"""Actual-device gates for packing, split reductions, reuse and CPU fallback."""
import numpy as np
import pytest
from lpsd_fast import api,lpsd
from lpsd._helpers import _kaiser_alpha

cp=pytest.importorskip('cupy')
try: DEVICES=cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError: DEVICES=0
pytestmark=pytest.mark.skipif(not DEVICES,reason='Optional real CUDA device required')


@pytest.fixture(scope='module')
def gpu():
    from benchmarks.cuda_packed import PackedCUDAEstimator
    return PackedCUDAEstimator(workers=4,batch_mb=1,threshold=256,fma=True,split_length=128)


def row(gpu,n,length,count):
    overlap=1-(n-length)/length/(count-1) if count>1 else .5
    return gpu.prepare_frequency(n,length,4.371,count,overlap,np.kaiser,_kaiser_alpha(200.)*np.pi,0)


@pytest.mark.parametrize('count',[1,7,8,9,17])
@pytest.mark.parametrize('precision',['fp64','twofold'])
@pytest.mark.parametrize('signal',['white','dc_nanovolt','constant','departed_transient'])
def test_split_powers_and_initial_group_boundaries(gpu,count,precision,signal):
    from concurrent.futures import ThreadPoolExecutor
    from benchmarks.cuda_packed import PackedBatch
    api._native()
    length=1027;n=length*5 if count>1 else length
    x=np.random.default_rng(count).normal(size=n)
    if signal=='dc_nanovolt': x=10+1e-9*x
    if signal=='constant': x.fill(10)
    if signal=='departed_transient': x[0]=1e10
    original=x.copy()
    rows=[row(gpu,n,length,count)]
    cr,ci,starts,s2=rows[0]
    expected=np.empty(count)
    gpu.bridge.bench_reference(api._pointer(x),api._pointer(cr),api._pointer(ci),length,gpu.ip(starts),count,0,api._pointer(expected))
    batch=PackedBatch(gpu,rows,256,precision)
    batch.launch(cp.asarray(x))
    actual=batch.out.get()
    with ThreadPoolExecutor(max_workers=2) as pool: values=batch.finish(x,actual,pool,1.)
    np.testing.assert_allclose(actual,expected,rtol=2e-9,atol=float(expected.max(initial=0))*2e-13)
    reference=2*gpu.bridge.bench_aggregate(api._pointer(expected),count)/s2
    assert abs(values[0]-reference)<=2e-9*abs(reference)+float(expected[1:].max(initial=0))*2e-13/s2
    np.testing.assert_array_equal(x,original)
    if signal=='constant': assert np.all(actual==0)


def test_variable_frequency_packing_and_workspace_reuse(gpu):
    from concurrent.futures import ThreadPoolExecutor
    from benchmarks.cuda_packed import PackedBatch
    n=16385;x=np.random.default_rng(23).normal(size=n)
    workspace={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for lengths in ([17,259,4097],[4097,128],[259,17,1027]):
            rows=[]
            for length in lengths:
                count=int(np.floor((n-length)*4/length+1+.5))
                rows.append(gpu.prepare_frequency(n,length,4.371,count,.75,np.kaiser,_kaiser_alpha(200.)*np.pi,0))
            batch=PackedBatch(gpu,rows,256,'fp64',workspace)
            batch.launch(cp.asarray(x));powers=batch.out.get()
            actual=batch.finish(x,powers,pool,1.)
            expected=[]
            for cr,ci,starts,s2 in rows:
                p=np.empty(len(starts))
                gpu.bridge.bench_reference(api._pointer(x),api._pointer(cr),api._pointer(ci),len(cr),gpu.ip(starts),len(starts),0,api._pointer(p))
                expected.append(2*gpu.bridge.bench_aggregate(api._pointer(p),len(p))/s2)
            np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=1e-24)


def test_full_batched_call_and_dc_fallback(gpu):
    x=np.random.default_rng(8).normal(size=32769)
    reference=lpsd(x,sample_rate=1.,workers=4,kernel='fast',outputs=('psd','nsd'))
    actual=gpu(x)
    np.testing.assert_array_equal(actual.index,reference.index)
    np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-24)
    assert gpu.last_profile['batches']>1
    x=10+1e-9*x
    reference=lpsd(x,sample_rate=1.,workers=4,kernel='auto',outputs=('psd','nsd'))
    actual=gpu(x)
    assert actual.attrs['cuda_experiment']['executed']=='CPU auto residual fallback'
    np.testing.assert_array_equal(actual,reference)
    ramp=np.linspace(-10.,10.,len(x))+1e-9*np.random.default_rng(9).normal(size=len(x))
    actual=gpu(ramp,detrending_order=1,n_frequencies=32,n_averages=8)
    assert actual.attrs['cuda_experiment']['executed']=='CPU fallback'
    assert actual.attrs['cuda_experiment']['workers']==4


def test_resident_plan_changes_input_without_overwriting_saved_output(gpu):
    from benchmarks.cuda_packed import ResidentPackedPlan
    plan=ResidentPackedPlan(gpu,8193,n_frequencies=64,n_averages=16)
    try:
        x=np.random.default_rng(77).normal(size=8193)
        first=plan(x);saved=first.to_numpy().copy()
        x=np.random.default_rng(78).normal(size=8193)
        reference=lpsd(x,sample_rate=1.,n_frequencies=64,n_averages=16,workers=4,kernel='fast',outputs=('psd','nsd'))
        # A streaming call with the same adapter must leave the resident plan intact.
        gpu(x,n_frequencies=32,n_averages=8)
        actual=plan(x)
        np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-24)
        np.testing.assert_array_equal(first,saved)
        x=10+1e-9*x
        reference=lpsd(x,sample_rate=1.,n_frequencies=64,n_averages=16,workers=4,kernel='auto',outputs=('psd','nsd'))
        np.testing.assert_array_equal(plan(x),reference)
        with pytest.raises(ValueError,match='length'): plan(x[:-1])
    finally: plan.close()
    with pytest.raises(RuntimeError,match='closed'): plan(x)


@pytest.mark.parametrize('backend',['cufft','cufft_graph','vkfft_graph'])
@pytest.mark.parametrize('n',[1025,7919,32768])
def test_fft_pipeline_graph_replay_odd_sizes_and_dc(gpu,backend,n):
    if backend.startswith('vkfft'): pytest.importorskip('pyvkfft')
    from benchmarks.bench_gpu_fft import GPUPeriodogram
    from benchmarks.bench_fftw import lpsd_grid,one_sided_psd,aggregate_power,KaiserWindow
    plan=GPUPeriodogram(n,backend)
    provider=KaiserWindow()
    w=provider.generate(n,_kaiser_alpha(200.)*np.pi)
    s2=provider.sums(w)[1]
    for scale,offset in [(1.,0.),(1e-9,10.),(0.,10.)]:
        x=offset+scale*np.random.default_rng(n).normal(size=n)
        centered=x-x[0];centered-=np.mean(centered)
        transformed=np.fft.rfft(centered*w)
        powers=one_sided_psd(transformed,n,1.,s2)
        expected=aggregate_power(powers,lpsd_grid(n,1.,200.,1000,100)['cuts'],1.,n)[0]
        actual=plan.compute(x)
        np.testing.assert_allclose(actual.psd,expected,rtol=2e-6,atol=1e-30)
        validation=plan.validation()
        assert validation['finite']
        assert validation['parseval_absolute_error']<=1e-11*max(np.max(x*x),1e-20)
        np.testing.assert_array_equal(x,offset+scale*np.random.default_rng(n).normal(size=n))
