"""Optional actual-device numerical gates for the isolated FP64 experiment."""
import ctypes as ct
import numpy as np
import pytest
from lpsd_fast import api
from lpsd._helpers import _kaiser_alpha

cp=pytest.importorskip('cupy',reason='Optional CUDA experiment requires CuPy')
try:
    DEVICES=cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    DEVICES=0
pytestmark=pytest.mark.skipif(not DEVICES,reason='No CUDA device available')


@pytest.fixture(scope='module')
def gpu():
    from benchmarks.cuda_lpsd import CUDAEstimator
    return CUDAEstimator()


@pytest.mark.parametrize('length',[17,128,259,1027,4097,65537])
@pytest.mark.parametrize('strategy,threads',[(s,t) for s in ('warp','block') for t in (64,128,256)])
@pytest.mark.parametrize('signal',['white','dc_nanovolt','constant','departed_transient'])
def test_actual_gpu_segment_powers_and_inherited_aggregation(gpu,length,strategy,threads,signal):
    api._native()
    n=length+int(32*(length/4+.125))
    count=int(np.floor((n-length)*4./length+1.+.5))
    x=np.random.default_rng(length).normal(size=n)
    if signal=='dc_nanovolt': x=10.+1e-9*x
    if signal=='constant': x.fill(10.)
    if signal=='departed_transient': x[0]=1e10
    original=x.copy()
    cr,ci,starts,_=gpu.prepare_frequency(n,length,4.371,count,.75,np.kaiser,_kaiser_alpha(200.)*np.pi,0)
    reference=np.empty(count)
    gpu.bridge.bench_reference(api._pointer(x),api._pointer(cr),api._pointer(ci),length,gpu.ip(starts),count,0,api._pointer(reference))
    actual=gpu.project(cp.asarray(x),cr,ci,starts,0,strategy,threads).get()
    # FP64 reductions change rounding. Compare each segment, including tails,
    # before replacing the cancellation-sensitive first CPU projections.
    np.testing.assert_allclose(actual,reference,rtol=2e-10,atol=float(reference.max(initial=0))*2e-14)
    gpu.bridge.bench_initial(api._pointer(x),api._pointer(cr),api._pointer(ci),length,gpu.ip(starts),count,0,api._pointer(actual))
    a=gpu.bridge.bench_aggregate(api._pointer(actual),count)
    b=gpu.bridge.bench_aggregate(api._pointer(reference),count)
    assert abs(a-b)<=2e-10*abs(b)+float(reference[1:].max(initial=0))*2e-14
    assert x.tobytes()==original.tobytes()
    if signal=='constant': assert np.all(actual==0)


def test_gpu_starts_keep_repeated_addition_rounding(gpu):
    n,length,overlap=7919,259,.73
    count=int(np.floor((n-length)/(1-overlap)/length+1.+.5))
    starts=np.empty(count,dtype=np.int32)
    assert gpu.bridge.bench_starts(gpu.ip(starts),n,length,overlap*100)==count
    start=0.
    expected=[]
    shift=(n-length)/(count-1)
    for _ in range(count):
        expected.append(int(np.floor(start+.5)))
        start+=shift
    np.testing.assert_array_equal(starts,expected)


def test_gpu_linear_detrending_reports_explicit_cpu_fallback(gpu):
    x=np.linspace(-10.,10.,1025)+1e-9*np.random.default_rng(9).normal(size=1025)
    result=gpu(x,detrending_order=1,n_frequencies=32,n_averages=8)
    assert result.attrs['cuda_experiment']['executed']=='CPU fallback'
