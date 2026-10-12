# Optimierte FFTW-Verfahren für PSD und NSD

Diese Erweiterung reduziert wiederholte Vorbereitung, Verwaltungsaufwand und
Speicherzugriffe in den beiden FFTW-Vergleichsverfahren. Die Definition des
jeweiligen Schätzers bleibt erhalten. Der an LPSD angepasste FFTW-Hybrid bleibt
eine Annäherung an LPSD/LNSD; die Optimierung macht daraus keinen identischen
Schätzer.

Die Ausgangsbasis ist Commit `61654be`. Der einfache FFTW-Vergleich ist in
[bench_fftw.py](../benchmarks/bench_fftw.py) implementiert. Für den bisherigen
Hybrid enthält
[matched_smoothing_20261010.py](../benchmarks/reference/matched_smoothing_20261010.py)
eine getrennte Referenzkopie. Der optimierte Hybrid liegt in
[matched_smoothing.py](../benchmarks/matched_smoothing.py).

Dieses Dokument enthält das festgelegte Messprofil, die abgeschlossenen
Vorher/Nachher-Messungen und ihre numerischen Grenzen. Bei einer Million
Proben benötigt der wiederverwendete Hybrid für NSD 28,583 statt 83,309 ms;
der vollständige optimierte Einzelaufruf benötigt 132,558 statt 260,270 ms.
Die folgenden Tabellen trennen diese Einsatzfälle und enthalten auch die
Primzahllänge sowie 30 Millionen Proben. Das gemeinsame Cachelimit von 2 GiB
begrenzt ausschließlich die Nutzdaten beider dauerhaften Caches, nicht den
gesamten Speicherbedarf des Prozesses.

## Vergleichsverfahren und unveränderte Größen

| Verfahren | Berechnung | Verhältnis zu LPSD |
|---|---|---|
| LPSD/LNSD mit `kernel="fast"` | Frequenzabhängige Segmente, periodisches Kaiserfenster und Mittelentfernung pro Segment | Referenz für Raster, Rauschstreuung und Linienform |
| Einfaches FFTW | Ein vollständiges Kaiserfenster, globale Mittelentfernung, einseitiges Periodogramm, Mittelwerte zusammenhängender FFT-Bin-Gruppen | Gleiche Ausgabefrequenzen, andere spektrale Mittelung |
| Angepasstes FFTW | Vollständiger Datensatz mit periodischem Tukeyfenster, Kaiser-Leistungsgewichte im Frequenzbereich und explizite native LPSD-Teilberechnung | Auf ähnliche Linienbreiten und Rauschstreuung ausgelegter Hybrid |

Die Vergleichseinstellungen sind `sample_rate=50`, `psll=200`,
`n_frequencies=1000`, `n_averages=100` und Mittelentfernung der Ordnung null.
`n_frequencies` ist die angeforderte Punktzahl; die tatsächlich verwendeten
Frequenzen liefert unverändert der LPSD-Planer. Die Datenlänge bestimmt auch
Segmentlängen, Segmentanzahlen und den Umfang der LPSD-Teilberechnung. Das
Messprotokoll muss deshalb die tatsächliche Punktzahl und die Grenzfrequenzen
enthalten, nicht nur die angeforderte Punktzahl.

PSD hat bei Eingangsdaten in Volt die Einheit V²/Hz, NSD die Einheit V/√Hz.
Die öffentliche Float32-Ausgabe und ihre Reihenfolge bleiben erhalten: Zuerst
wird die PSD gerundet, anschließend wird die NSD auf dem bisherigen
Complex64-Wurzelpfad daraus berechnet. NSD ist somit die Wurzel der gemittelten
Leistung, nicht der Mittelwert einzelner Amplituden. Eine gemeinsame Anforderung
`outputs=("psd", "nsd")` nutzt eine einzige Spektrumberechnung.

## Festgelegtes Messprofil

Das Profil verwendet für das einfache FFTW `operations="native_grouped"`.
Der Hybrid nutzt native Vor-/Nachverarbeitung und native Kaiser-Glättung,
`weight_dtype="float32"`, ein Gewichtslimit von 1536 MiB und ein gemeinsames
Gewichts-/LPSD-Koeffizientenlimit von 2048 MiB. Das nach tatsächlich belegten
Gewichten verbleibende Budget steht automatisch für LPSD-Koeffizienten zur
Verfügung. Die aufgezeichneten Nutzdaten können deshalb unter dem Limit
liegen; die Aufteilung ist keine feste Reservierung von 1536 plus 512 MiB.

Die vorbereitbare Hybridklasse verwendet vier Glättungsworker und bis zu acht
LPSD-Worker. `estimate_once` verwendet im Vergleich acht Glättungsworker, da
alle Kernel ohne dauerhaften Gewichtscache berechnet werden. Die interne
FFTW-Threadzahl wird unabhängig davon nach Datenlänge gewählt:

| Eingangslänge N | Optimierte FFTW-Threads | Optimierte DFT | Planer für Reuse nachher | Planer für alle vollständigen Neuaufrufe |
|---|---:|---|---|---|
| 100 000 | 1 | Direktes R2C | ESTIMATE | ESTIMATE |
| 1 000 000 | 4 | Direktes R2C | ESTIMATE | ESTIMATE |
| 1 000 003 | 8 | Bluestein, interne Faltung M = 1 512 000 | MEASURE, FFTW-Zeitlimit 1 s je Plan | ESTIMATE |
| 10 000 000 | 4 | Direktes R2C | ESTIMATE | ESTIMATE |
| 30 000 000 | 8 | Direktes R2C | ESTIMATE | ESTIMATE |

Die bisherigen FFTW-Verfahren verwenden weiterhin acht FFTW-Threads und
ESTIMATE. Auch der vollständige LPSD/LNSD-Referenzaufruf verwendet acht Worker.
Die gemessenen Vorher/Nachher-Faktoren bewerten dieses gesamte Profil
einschließlich Threadzahl, FFT-Algorithmus und Planerauswahl. Die Wirkung
einzelner Änderungen muss anhand gesonderter Messungen beurteilt werden.

MEASURE ist ausschließlich für die wiederverwendeten optimierten Objekte bei
N = 1 000 003 ausgewählt. Die beobachtete Vorbereitungszeit wird separat
protokolliert und gehört nicht zur Reuse-Berechnungszeit. Das FFTW-Zeitlimit
gilt für jeden Plan, nicht für den gesamten Objektaufbau; Bluestein benötigt
zwei komplexe Faltungspläne. Der Kaltstart der vorbereitbaren Klasse und der
optimierte Einzelaufruf verwenden ausdrücklich ESTIMATE, auch bei dieser
Primzahllänge. Die Einstellung erfolgt über `--measure-lengths` und
`--planner-time-limit`; `--fftw-threads N:T` legt die interne Threadzahl fest.

## Abgeschlossene Messung vom 10. Oktober 2026

Die folgenden Mediane stammen aus `results_final.json`, abgeschlossen am
10. Oktober 2026 um 19:25:58 UTC. Für 100 000, 1 000 000 und 1 000 003 Proben
wurden neun Wiederholungen je Variante gemessen, für 10 und 30 Millionen
Proben jeweils drei. Alle Tabellenwerte sind Millisekunden; der HTML-Bericht
enthält zusätzlich jede Minimum–Maximum-Spanne und sämtliche Rohzeiten.

Der Rechner stellte AMD EPYC 9V74, acht CPU-Äquivalente als Container-Kontingent
und 8 GiB Speicher bereit. Verwendet wurden Python 3.12.14, NumPy 2.3.5 und
FFTW 3.3.11 mit SSE2/AVX/AVX2/AVX-512-Unterstützung. Die Angaben beschreiben
diesen virtuellen Host und sind keine Laufzeitgarantie für andere Systeme.

### Einfaches FFTW: vollständige Aufrufe

| N | Ausgabe | Reuse vorher [ms] | Reuse optimiert [ms] | Komplett neu vorher [ms] | Komplett neu optimiert [ms] |
|---:|---|---:|---:|---:|---:|
| 100 000 | PSD | 0,547 | 0,409 | 22,391 | 8,024 |
| 100 000 | NSD | 0,575 | 0,424 | 22,399 | 8,103 |
| 1 000 000 | PSD | 5,260 | 4,352 | 42,024 | 32,212 |
| 1 000 000 | NSD | 5,943 | 4,510 | 40,766 | 31,262 |
| 1 000 003 | PSD | 78,685 | 24,537 | 251,685 | 120,117 |
| 1 000 003 | NSD | 79,864 | 26,787 | 252,813 | 120,191 |
| 10 000 000 | PSD | 68,285 | 58,819 | 241,424 | 220,742 |
| 10 000 000 | NSD | 72,631 | 65,132 | 236,601 | 207,904 |
| 30 000 000 | PSD | 251,700 | 220,364 | 743,146 | 673,512 |
| 30 000 000 | NSD | 257,646 | 205,106 | 743,554 | 696,366 |

Bei einer Million Proben sinkt die NSD-Reuse-Zeit von 5,943 auf 4,510 ms,
der vollständige Neuaufruf von 40,766 auf 31,262 ms. Der größere Gewinn bei
N = 1 000 003 enthält ausdrücklich den anderen DFT-Pfad und die MEASURE-
Vorbereitung der wiederverwendeten optimierten Instanz.

### Angepasster FFTW-Hybrid: Reuse, Klassenaufbau und Einzelaufruf

| N | Ausgabe | Reuse vorher [ms] | Reuse optimiert [ms] | Neu vorher [ms] | Kaltstart Klasse [ms] | Einmalig optimiert [ms] | LPSD/LNSD komplett [ms] |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 000 | PSD | 30,373 | 5,948 | 80,612 | 62,141 | 46,200 | 87,192 |
| 100 000 | NSD | 25,101 | 6,558 | 84,699 | 58,565 | 43,074 | 92,910 |
| 1 000 000 | PSD | 87,728 | 31,700 | 256,874 | 206,808 | 131,477 | 149,360 |
| 1 000 000 | NSD | 83,309 | 28,583 | 260,270 | 198,362 | 132,558 | 150,495 |
| 1 000 003 | PSD | 129,568 | 48,234 | 439,995 | 241,218 | 209,827 | 139,203 |
| 1 000 003 | NSD | 127,216 | 48,190 | 430,286 | 238,683 | 201,341 | 138,014 |
| 10 000 000 | PSD | 1399,030 | 513,012 | 2842,431 | 1434,672 | 1515,374 | 1581,234 |
| 10 000 000 | NSD | 1329,680 | 557,326 | 2699,205 | 1511,455 | 1207,415 | 1647,829 |
| 30 000 000 | PSD | 5088,372 | 2231,869 | 9648,202 | 4978,291 | 4300,619 | 4909,269 |
| 30 000 000 | NSD | 5669,993 | 2221,787 | 9791,502 | 4734,558 | 4192,119 | 5921,147 |

Bei einer Million Proben ist der wiederverwendete Hybrid für NSD 2,91-mal
schneller als seine vorherige Implementierung: 83,309 gegenüber 28,583 ms.
Der vollständige optimierte Einzelaufruf benötigt 132,558 statt 260,270 ms,
also etwa die halbe Zeit. Der vollständige LNSD-Referenzaufruf benötigt dort
150,495 ms. Bei 30 Millionen Proben sinken die NSD-Zeiten für Reuse von
5 669,993 auf 2 221,787 ms und für den einmaligen vollständigen Aufruf von
9 791,502 auf 4 192,119 ms; der vollständige LNSD-Aufruf liegt bei 5 921,147 ms.

Die Primzahllänge verlangt eine andere Einordnung: Der Hybrid erreicht
48,190 ms mit Wiederverwendung, benötigt für den vollständigen optimierten
Einzelaufruf aber 201,341 ms gegenüber 138,014 ms für LNSD. Auch sein Kaltstart
mit vorbereitbaren Caches liegt mit 238,683 ms darüber. Die einmal beobachtete
Vorbereitung der wiederverwendeten Hybridinstanz benötigt 2 196,901 ms;
2 018,146 ms davon entfallen auf die beiden FFTW-Pläne. Die Reuse-Zahl darf
diese anfänglichen Kosten nicht verdecken.

Die Zeitstreuung bleibt sichtbar: Bei einer Million Proben reichen die
NSD-Reuse-Aufrufe des optimierten Hybrids von 26,215 bis 67,831 ms. Bei
30 Millionen liegen sie zwischen 2 076,312 und 2 382,821 ms. Die einzelnen
PSD-/NSD-Mediane stammen aus getrennten Aufrufen desselben weitgehend
gemeinsamen Rechenwegs; ihre Differenz belegt keinen grundsätzlichen
Geschwindigkeitsvorteil einer Ausgabedarstellung.

### Gemeinsame PSD- und NSD-Ausgabe

| N | Einfach, Reuse vorher → nachher [ms] | Einfach, neu vorher → nachher [ms] | Hybrid, Reuse vorher → nachher [ms] | Hybrid, neu vorher → Kaltstart Klasse [ms] | Hybrid, einmalig optimiert [ms] |
|---:|---:|---:|---:|---:|---:|
| 100 000 | 1,375 → 0,437 | 22,864 → 8,159 | 30,215 → 6,975 | 78,623 → 60,628 | 47,717 |
| 1 000 000 | 10,296 → 4,255 | 49,157 → 32,554 | 80,440 → 32,623 | 251,724 → 192,933 | 141,859 |
| 1 000 003 | 167,624 → 27,098 | 324,458 → 115,289 | 132,032 → 49,114 | 442,783 → 243,761 | 204,048 |
| 10 000 000 | 136,327 → 54,407 | 316,283 → 236,376 | 1562,176 → 661,012 | 2875,778 → 1690,863 | 1242,454 |
| 30 000 000 | 530,870 → 209,275 | 1028,966 → 692,878 | 5705,191 → 2095,901 | 9955,983 → 4760,601 | 3486,177 |

Das einfache Ausgangsverfahren berechnet hierfür zweimal; die optimierte
API teilt sich eine FFT und eine Leistungsauswertung. Der Ausgangshybrid
unterstützte die gemeinsame Ausgabe bereits. Die verschiedenen Gewinne
dieser Zeilen dürfen deshalb nicht ausschließlich dem FFT-Kern zugeschrieben
werden.

### Gemessene Gleichheit und Annäherung

Die sechs Qualitätssignale der Hauptmessung haben jeweils N = 1 000 000
und 577 unveränderte Ausgabefrequenzen. Beim einfachen FFTW und beim
optimierten Einzelaufruf des Hybrids sind sämtliche öffentlichen PSD- und
NSD-Werte bitidentisch mit der jeweiligen vorherigen Implementierung.
Die vorbereitbare Hybridklasse mit Float32-Gewichten verändert einige
gerundete Werte: Die größte relative Änderung über alle sechs Signale beträgt
1,172 × 10⁻⁷ für PSD und 1,125 × 10⁻⁷ für NSD. Diese Aussage betrifft den
Implementierungsvergleich bei dieser Datenlänge; sie ist keine allgemeine
Fehlergarantie und keine Gleichheitsbehauptung gegenüber LPSD.

Für weißes Rauschen ist die beschreibende Standardabweichung von
NSD/√PSD des Quellmodells über 0,05–20 Hz bei LNSD 2,6393 %, beim einfachen
FFTW 10,9020 % und beim optimierten Hybrid 2,5075 %. Beim Hybrid vorher
beträgt sie ebenfalls 2,5075 % auf diesen angegebenen Stellen. Die tiefe
LPSD-Teilberechnung endet bei 0,0397304 Hz; die Statistik erfasst damit
vollständig FFTW-berechnete Punkte. Die ähnlichen Streuungen verlangen
keine identischen Zufallszacken an jeder Frequenz. Eine neue unabhängige
Monte-Carlo-Studie gehört nicht zu diesem Optimierungslauf.

Der isolierte Ton bei 3,1230185 Hz liefert folgende Werte. Die FWHM wird aus
den vorhandenen Stützstellen linear interpoliert; zusätzliche Dezimalstellen
dienen der Reproduzierbarkeit, nicht einer höheren Frequenzauflösung.

| Methode | PSD-FWHM [Hz] | NSD-FWHM [Hz] | Maximale ferne NSD [fV/√Hz] |
|---|---:|---:|---:|
| LPSD/LNSD | 0,115707809 | 0,159071171 | 0,017537026 |
| Einfaches FFTW, optimiert | 0,041052674 | 0,041052674 | 0,016156619 |
| Hybrid, vorher | 0,115707952 | 0,159071400 | 1,297458599 |
| Hybrid, optimiert | 0,115707952 | 0,159071400 | 1,297458599 |

Der ferne Bereich ist 0,01 < f < 20 Hz und |f − f₀| > 1 Hz. Der höhere
ferne Leckboden des Hybrids ist bereits im ursprünglichen Schätzer vorhanden;
die Implementierungsoptimierung verändert ihn nicht auf den angegebenen
Stellen. Eine passende Hauptkeulenbreite bedeutet deshalb nicht gleiche
Nebenkeulen wie LPSD.

Die gesonderte Datei `quality_prime_final.json` prüft dieselben sechs
Signaltypen bei N = 1 000 003 mit Bluestein. Beim einfachen FFTW und bei
`estimate_once` bleiben jeweils fünf der sechs Signale bitidentisch; nur
der reine Sinus zeigt Änderungen. Beim einfachen FFTW liegt dessen größter
punktweiser relativer PSD-Fehler bei 0,093532 %, der NSD-Fehler bei
0,046755 %. Gleichzeitig betragen die größten absoluten Fehler nur
7,146 × 10⁻⁴⁰ V²/Hz und 2,256 × 10⁻²² V/√Hz. Diese relativen Maxima
entstehen an nahezu verschwindenden Leckwerten und bedeuten keine entsprechend
große Veränderung der Peakform.

Der vorbereitete Hybrid erreicht bei dieser Primzahllänge maximal
3,816 × 10⁻⁶ relative PSD-Änderung und 1,889 × 10⁻⁶ relative NSD-Änderung.
Die jeweils größte absolute Änderung, auf das Maximum der zugehörigen
Referenz bezogen, bleibt über alle Signale bei höchstens 9,125 × 10⁻⁸
beziehungsweise 7,872 × 10⁻⁸. Alle Prüfungen verwenden gemeinsam
`rtol=3e-6` und `atol=max(Referenzmaximum*2e-14, 2*float32.tiny)`;
die absoluten und relativen Rohfehler bleiben zusätzlich aufgezeichnet.

### Regressionsstatus

Der abschließende vollständige Pytest-Lauf meldet **1 276 bestanden,
einen bestehenden CUDA/CuPy-Skip und zwei bestehende Xfails**. Es gibt keine
neuen Skips für FFTW, Hybrid oder vorbereitete LPSD-Teilmengen. Die verwendeten
Quell- und Native-Dateien entsprechen weiterhin den 16 im Haupt-JSON
dokumentierten Hashes. Portable Build und Wheel-Smoke werden im zugehörigen
Abschlussprotokoll getrennt von den wissenschaftlichen Messreihen geführt.

## Gemeinsame native Vor- und Nachverarbeitung

[FFTWOperations](../benchmarks/_fftw_native.py) und die zugehörigen
[C-Funktionen](../lpsd_fast/_native/fftw_ops.h) verwenden vorbereitete, private
Puffer. Deren Formen, Datentypen und Speicheranordnung werden beim Binden
geprüft; Zeiger und konstante Normierungsgrößen werden danach wiederverwendet.
Die öffentlichen Berechnungsaufrufe prüfen neue Eingangsdaten weiterhin. Der
interne Pfad für bereits geprüfte Daten vermeidet lediglich einen zweiten
vollständigen Prüfdurchlauf innerhalb derselben Pipeline.

Die numerisch wichtige Verankerung der Mittelentfernung bleibt bestehen:

1. `samples - samples[0]` wird wie bisher mit NumPy in den Zeitpuffer geschrieben.
2. `numpy.mean` bestimmt unverändert den Mittelwert dieser verankerten Werte.
3. Ein nativer Durchlauf zieht diesen Mittelwert ab und multipliziert mit dem
   unveränderten Fenster.

Damit wird kein großer Gleichanteil nachträglich von einer ähnlich großen
Fouriersumme abgezogen. Der Prüfdatensatz mit 10 V Gleichspannung und
Nanovolt-Rauschen prüft gerade diesen empfindlichen Fall.

Nach der FFT berechnet ein weiterer nativer Durchlauf Realteilquadrat,
Imaginärteilquadrat und Normierung gemeinsam. Für die unnormierte N-Punkt-DFT
gilt

\[
P_k = \frac{c_k |X_k|^2}{f_s\sum_n w_n^2},
\qquad
c_k=\begin{cases}
1,& k=0\text{ oder geradzahliges }N\text{ und }k=N/2,\\
2,& \text{sonst}.
\end{cases}
\]

Insbesondere bleibt bei ungeradem N der letzte ausgegebene FFT-Bin verdoppelt.
Der native Pfad benötigt für diese Berechnung keinen zweiten vollständigen
Leistungs-Zwischenpuffer. `operations="numpy"` behält den ursprünglichen
Array-Pfad als Vergleich; `auto` protokolliert, welches Backend tatsächlich
verwendet wurde.

### Direkte logarithmische Gruppensummen

Beim einfachen FFTW-Verfahren wählt `operations="native_grouped"` zusätzlich
direkte Gruppensummen aus der komplexen FFT-Ausgabe. Die Gruppen enthalten
dieselben positiven FFT-Bins wie zuvor. Ihre Grenzen stammen weiterhin aus den
Mittelpunkten zwischen den LPSD-Frequenzlabels; DC gehört nicht zur positiven
Partition. Für eine Gruppe \(G_j\) mit \(C_j\) Bins bleiben

\[
\widehat S_j=\frac{1}{C_j}\sum_{k\in G_j}P_k,
\qquad
E_j=\frac{f_s}{N}\sum_{k\in G_j}P_k
\]

die ausgegebene Dichte beziehungsweise die integrierte Bandleistung.

Die native Schleife verbindet Leistungsbildung, Endpunktbehandlung,
Normierung und Gruppensumme. Ein vollständiges Array aller \(P_k\) muss dafür
nicht materialisiert werden. Die SIMD-Summation kann gegenüber
`numpy.add.reduceat` anders runden; weder die Bin-Zuordnung noch die
mathematische Normierung wird geändert. Für eine unabhängige
Normierungskontrolle darf der Prüfcode das vollständige Leistungsarray
außerhalb der Zeitmessung rekonstruieren.

## Vorbereitete LPSD-Teilberechnung

[PreparedLPSDSubset](../lpsd_fast/prepared.py) übernimmt einen unveränderten
Plan `(f, r, m, lengths, counts)`. Eine Teilmengenauswahl muss auf alle fünf
Arrays gleichermaßen angewendet werden. Die Klasse bereitet pro ausgewählter
Frequenz das Kaiserfenster, die Normierungen und die projizierten
Fourierkoeffizienten vor.

Für Fensterkoeffizienten \(c_n\) und Segmentlänge \(L\) ist die Projektion

\[
q_n=c_n-\frac{1}{L}\sum_r c_r.
\]

In exakter Arithmetik entspricht die segmentweise Mittelentfernung damit einer
Summe mit \(q_n(x_{s+n}-x_s)\). Der bestehende native `fast`-Kern verwendet
weiterhin seine lokale Verankerung, Segmentstartpunkte und Reihenfolge der
Leistungsaktualisierungen. Vorbereitet werden datenunabhängige Koeffizienten;
die Segmente selbst werden bei jedem Aufruf erneut ausgewertet.

Der Cache enthält schreibgeschützte Float64-Vektoren für Real- und Imaginärteil
von \(q\), dazu die kleinen Normierungsdaten. Die Vektoren benötigen
\(16L\) Bytes je vollständig gespeicherter Frequenz. Punkte werden in
Planreihenfolge aufgenommen, sofern sie in `max_cache_mb` passen. Nicht
gespeicherte Punkte werden während des jeweiligen Aufrufs vorbereitet. Weder
Eingangsdaten noch bereits berechnete Spektren werden als Ergebnisersatz
gespeichert.

Unabhängige Frequenzen werden vorab zu höchstens zweimal so vielen Aufgaben wie
verfügbaren Workern gebündelt. Eine Kostenschätzung aus Segmentlänge und
Segmentanzahl verteilt die Arbeit und berücksichtigt zusätzliche Vorbereitung
bei fehlenden Cacheeinträgen. Diese Schätzung dient nur der Arbeitsverteilung.
Sie ist kein gemessener Zeitbedarf und verändert keine Summationsreihenfolge
innerhalb einer Frequenz. Der Threadpool bleibt für Wiederholungsaufrufe
bestehen.

`compute(values)` liefert ein neues Float32-PSD-Array in der Reihenfolge des
übergebenen Plans. `close()` beziehungsweise ein Kontextmanager gibt Cache und
Worker frei. Die bestehende öffentliche LPSD/LNSD-API wird durch diese additive
Klasse nicht ersetzt.

## Native Kaiser-Glättung des Hybrids

Der Hybrid verwendet weiterhin das periodische Tukeyfenster mit einem
Taper-Anteil von 0,05 für die vollständige FFT. Die frequenzabhängigen
Leistungsgewichte stammen aus derselben analytischen Kaiserantwort wie in der
Referenzkopie. Ihre Skalierung verwendet die ursprünglichen LPSD-Segmentlängen.

Unverändert bleiben die Kaiserformel, ihr bisheriger Stützbereich, die
reflektierten Beiträge und die diskrete Normierung. Auch die halben
Quadraturgewichte für DC und den vorhandenen Nyquist-Endpunkt im Nenner
bleiben erhalten. Es werden weder Kernelhälften weggelassen noch schmalere
Fenster als Geschwindigkeitsabkürzung verwendet. Die Implementierung liegt in
[fftw_smoothing.h](../lpsd_fast/_native/fftw_smoothing.h).

Die gespeicherte Variante berechnet normalisierte Gewichte einmal und führt
anschließend native Skalarprodukte mit dem Float64-Leistungsspektrum aus.
Passt ein Kernel nicht in `max_kernel_cache_mb`, erzeugt die native Schleife
seine Gewichte während der Akkumulation. Dafür werden keine temporären
Gewichtsarrays angelegt. Die Formel wird direkt ausgewertet; eine
Interpolations- oder Nachschlagetabelle ist nicht Bestandteil dieses Pfads.
Der explizite NumPy-Pfad bleibt als unabhängige Formelimplementierung erhalten.

Die LPSD-Auswahl des Hybrids bleibt ebenfalls unverändert:

```python
low_mask = (lengths > n * (1 / 16)) | (lengths < 32)
```

Die erste Bedingung erfasst die langen Segmente im tiefen Frequenzbereich;
die zweite nimmt sehr kurze Segmente unabhängig davon von der kontinuierlichen
Fensterannäherung aus. Diese Punkte werden mit `PreparedLPSDSubset` berechnet.
Ihre Kosten gehören vollständig zum jeweiligen Hybridaufruf. Berichte müssen
den tatsächlichen Auswahlbereich kennzeichnen; eine über alle Frequenzen
berechnete Ähnlichkeitsstatistik kann sonst durch die identischen
LPSD-Teilpunkte günstiger erscheinen.

Die als `enbw` gespeicherte Größe beschreibt die Zielbandbreite des
Kaiserfensters. Sie ist keine exakt bestimmte ENBW der zusammengesetzten
Tukey-FFT mit nachfolgender Glättung. Der Hybrid übernimmt auch nicht die
200-dB-Nebenkeuleneigenschaft des Kaiser-Zielfensters. Ähnliche Hauptkeulen
können mit einem anderen fernen Leckboden einhergehen; deshalb gehört dieser
zum Vergleich.

### Optionaler Float32-Gewichtscache

`weight_dtype="float32"` verändert ausschließlich die gespeicherten,
normalisierten Glättungsgewichte. Die Gewichte werden zunächst in Float64
erzeugt und normiert und erst danach einmal in den schmaleren Cache kopiert.
Eingangsdaten, Zeitfenster, komplexe FFT, Leistungsspektrum, Akkumulation und
die projizierten LPSD-Koeffizienten bleiben in doppelter Genauigkeit.

Ein gespeichertes Gewicht benötigt damit vier statt acht Bytes. Derselbe
Nutzdaten-Cache kann mehr Kernel aufnehmen und bei Wiederholungen weniger
Daten übertragen. Während der Vorbereitung benötigt ein bearbeiteter Kernel
zusätzlich seinen Float64-Zwischenpuffer; der dauerhafte Cacheumfang allein
beschreibt diesen Spitzenbedarf nicht.

`weight_dtype="float64"` bleibt die numerische Vergleichsvariante. Die
Float32-Gewichte können gerundete Ergebniswerte verändern und werden deshalb
gegen denselben bisherigen Hybrid geprüft. Eine Übereinstimmung auf den
Testsignalen ist keine allgemeine relative Fehlergarantie für spektrale Nullen.

## Einmaliger Aufruf und aufgeschobene Normierung

Ein großer dauerhafter Cache lohnt sich nur, wenn ein Objekt wiederverwendet
wird. Für einen vollständigen Einzelaufruf stellt der Hybrid deshalb
`estimate_once(backend, samples, outputs="psd", **kwargs)` bereit. Diese
Funktion erzwingt sowohl `max_kernel_cache_mb=0` als auch `window_cache_mb=0`,
erzeugt das Objekt, berechnet das Spektrum und schließt es anschließend.
FFT-Puffer, Fenster und notwendiger temporärer Arbeitsraum entstehen weiterhin.

Zusätzlich aktiviert der native Einzelaufruf standardmäßig
`defer_normalization=True`. Für nicht gespeicherte Glättungskernel wird dann
im Konstruktor kein separater Formeldurchlauf allein zur Nennerberechnung
ausgeführt. Der erste Berechnungsdurchlauf akkumuliert Zähler und Nenner
gemeinsam. Bei späterer Wiederverwendung desselben Objekts bleiben die kleinen
Nennerwerte verfügbar. Gespeicherte Kernel behalten ihre bisherige
Vorbereitung; der NumPy-Rückfallpfad berechnet den Nenner weiterhin vorab.
Die Metadaten unterscheiden angefordertes und tatsächlich verwendetes
Verhalten.

Die vollständige Zeit für `estimate_once` schließt Planung, Fenster,
Berechnung und Freigabe ein. Die Funktion löscht FFTW-Wisdom nicht selbst.
Ein Vergleich mit ausdrücklich kalter FFTW-Planung muss den Wisdom-Zustand
außerhalb beziehungsweise als dokumentierten Teil seiner Messhülle einstellen.

## Optionaler Bluestein-Pfad

[BluesteinRealFFT](../benchmarks/_fftw_bluestein.py) berechnet ausdrücklich die
ursprüngliche N-Punkt-DFT. Es werden keine neuen Eingangsproben angehängt, um
anschließend eine andere DFT als Spektrum auszugeben. Ausgangsraster
\(f_k=kf_s/N\), Normierung, Fenster und Schätzer bleiben erhalten.

Bluestein formuliert die DFT als eine komplexe Faltung mit Chirpfaktoren. Der
Adapter bereitet die Fouriertransformierte des Faltungskerns vor; jeder
Berechnungsaufruf benötigt eine vorwärts und eine rückwärts laufende komplexe
FFTW-Transformation. Die inverse Faltung wird durch ihre interne Länge
\(M\) normiert, sodass die ausgegebene N-Punkt-DFT wie bisher unnormiert ist.
Die Rundungsfehler können sich von einem direkten FFTW-R2C-Plan unterscheiden.
„Exakt“ bezeichnet hier dieselbe mathematische DFT, keine bitweise Identität.

Standardmäßig verwendet der Adapter
`next_power_of_two(2*N - 1)`. Werden nur die für reelle Eingangsdaten benötigten
\(K=\lfloor N/2\rfloor+1\) Ausgabebins berechnet, genügen die Faltungslags
\(-(N-1),\ldots,K-1\). Eine explizite interne Faltungslänge darf deshalb bis
auf

\[
M\geq N+K-1=N+\lfloor N/2\rfloor
\]

verkürzt werden. Das betrifft ausschließlich die interne Faltung; es ist weder
eine Ausdünnung der Eingangsproben noch eine Änderung des Frequenzrasters.
Eine zulässige Länge ist nicht automatisch die schnellste Länge auf einer
bestimmten FFTW-Installation.

Der Adapter stellt dieselbe grundlegende `x`/`y`/`execute`/`close`-Schnittstelle
wie `RealFFT` bereit. Die gemeinsame Fabrik `create_real_fft` wählt ausdrücklich
`algorithm="native"` oder `algorithm="bluestein"`; `convolution_length` legt
optional die interne Länge fest. In `Periodogram` und `MatchedSmoothing` heißen
die entsprechenden Parameter `fft_algorithm` und `fft_convolution_length`.
`planner` und `time_limit` bestimmen den Planungsmodus und das optionale
FFTW-Zeitlimit je Plan. Das festgelegte Messprofil unterscheidet deren Auswahl
für wiederverwendete Objekte von der für vollständige Neuaufrufe.
Es gibt keine verdeckte Umschaltung anhand eines Spektralfehlers. Die
zusätzlichen [nativen Schleifen](../lpsd_fast/_native/fftw_bluestein_ops.h)
können Chirpmultiplikation und Ausgabenormierung zusammenfassen. Sie verändern
die DFT-Definition ebenfalls nicht. Die Pläne verwenden die dokumentierte
interne FFTW-Threadzahl.

Der FFTW-Planerzustand einschließlich Threadkonfiguration, Wisdom und Zeitlimit
ist global. Konstruktoren verschiedener FFTW-Objekte und Änderungen dieses
Zustands müssen seriell ausgeführt werden; parallele Konstruktoren sind nicht
unterstützt. Ein FFTW-Berechnungsobjekt ist nicht reentrant. Unabhängig
vorbereitete Instanzen können separat ausführen, sofern ihre jeweiligen Puffer
nicht gemeinsam verändert werden. Die Vergleichsmessung führt ihre Aufrufe
seriell aus.

Der zusätzliche Speicher für Chirp, vorbereiteten Faltungskern und komplexen
Arbeitsraum muss in einer Gesamtbetrachtung enthalten sein. Der Adapter
protokolliert bekannte Puffergrößen und schließt diese von FFTW-internem
Planspeicher begrifflich sauber ab. Ob Bluestein für eine bestimmte Datenlänge
zweckmäßig ist, entscheiden vollständige Zeit- und Speichervergleiche. Ein
langsamer direkter R2C-Plan bei einer Primzahllänge ist keine allgemeine Grenze
von FFTW.

## Messumfang und Reproduzierbarkeit

Der [Vergleichslauf](../benchmarks/bench_fftw_optimized.py) lädt die bisherige
Implementierung aus einem getrennten Checkout mit eigener nativer Bibliothek.
Commit, Quelltext-Hashes, Bibliotheks-Hashes, FFTW-Version, Rechnerdaten,
Threadkonfiguration und Einstellungen werden protokolliert. Ein Vergleich
„vorher/nachher“ bezieht sich immer auf dieselbe der beiden FFTW-Methoden.
Der zusätzliche Vergleich mit LPSD bewertet dagegen unterschiedliche Schätzer.
Die Skalierungszeiten werden am weißen Rauschen gemessen. Die sechs Signale
dienen der gesonderten Qualitätsprüfung; deren Spektren dürfen nicht als sechs
zusätzlich gemessene Laufzeitreihen beschriftet werden.

| Messart | Enthalten | Vor Beginn verfügbar |
|---|---|---|
| `reuse` | Vollständige eingangsabhängige Berechnung einschließlich Eingangsprüfung, Fensteranwendung, FFT, Dichte, Glättung/LPSD-Teilmenge und besitzender Ausgabe | Pläne nach Reuse-Profil, Puffer, Fenster, Raster und konfigurierte Caches; Vorbereitung separat gemessen |
| `fresh` | Konstruktion mit ESTIMATE und dem gewählten Cacheprofil, Berechnung und Freigabe; in der Vergleichshülle einschließlich Wisdom-Löschung | Geladene Bibliotheken und bereits erzeugte Eingangsdaten |
| `estimate_once` | Konstruktion mit ESTIMATE ohne dauerhafte Gewichts- oder LPSD-Koeffizientencaches, Berechnung mit aufgeschobener nativer Normierung, Freigabe | Geladene Bibliotheken und bereits erzeugte Eingangsdaten; Wisdom gemäß Messhülle |

`fresh` und `estimate_once` beantworten verschiedene Einsatzfragen. Ein mit
großen Caches vollständig neu aufgebautes Objekt ist nicht mit dem speziell
für eine einmalige Verwendung vorbereiteten Pfad gleichzusetzen. Als
Beschleunigung muss jeweils das Verhältnis vergleichbarer vollständiger
Aufrufe angegeben werden, nicht nur die Zeit des FFT-Kerns.

Im JSON-Protokoll steht der Einzelaufruf als `method="matched_once"` und
`mode="fresh"`. `matched_after` mit `mode="fresh"` bleibt der vollständige
Kaltstart der vorbereitbaren Klasse. Die Laufzeitplots und Tabellen zeigen
diese beiden vollständigen Neuaufrufe getrennt vom wiederverwendeten Objekt.
Die Genauigkeitswerte von `matched_once` werden zusätzlich gespeichert, auch
wenn dessen Spektralkurve nicht nochmals in der NPZ-Datei abgelegt wird.

Der kombinierte PSD/NSD-Aufruf ist gesondert auszuweisen: Die bisherige einfache
FFTW-API benötigt dafür zwei vollständige Aufrufe; die optimierte Variante
berechnet beides aus einem Spektrum. Der bisherige Hybrid unterstützt einen
kombinierten Aufruf bereits. Ein Vorteil aus dieser API-Änderung darf nicht
als Verdopplung der FFT-Rechengeschwindigkeit bezeichnet werden.

Signalerzeugung, Importe, Laden der Bibliotheken, Aufwärmaufrufe, externe
Ergebniskontrollen, Hashbildung, Dateiausgabe und Plotten gehören nicht zu den
Berechnungszeiten. Die Eingangsprüfung innerhalb der öffentlichen Berechnung
bleibt enthalten. Instrumentierte Phasenprofile werden getrennt gemessen;
Zeiten gleichzeitig arbeitender Worker sind nicht zu einer verstrichenen
Gesamtzeit addierbar. Die einfachen FFTW-Phasenprofile verwenden keinen
expliziten Ausgabeschalter: optimiert ist es die Standard-PSD, bei der
Referenz der zuletzt eingestellte Dichtemodus. Die Hybridprofile fordern
explizit NSD an. Die einfachen Profilsummen werden deshalb nicht als
NSD-Mediane beschriftet oder anstelle vollständiger Aufrufzeiten verwendet.

Die Verfahren laufen seriell in wechselnder, balancierter Reihenfolge.
Aufbewahrt werden alle Wiederholungen mit verstrichener Zeit und CPU-Zeit;
Diagramme zeigen Median sowie beobachtetes Minimum und Maximum. Die Spannweite
ist kein Konfidenzintervall. Taktänderungen und andere Nutzer des virtuellen
Hosts können Zeitunterschiede verursachen. PSD und NSD teilen fast ihre gesamte
Berechnung; unterschiedliche Mediane allein belegen deshalb keinen besonderen
Geschwindigkeitsvorteil einer Dichtedarstellung.

Die abgeschlossene Größenskala umfasst 100 000, 1 000 000, 1 000 003,
10 000 000 und 30 000 000 Proben. Die Hauptmessung ist abgeschlossen;
Zwischenläufe werden nicht mit diesen Reihen vermischt.

## Signale und Qualitätsprüfung

Die sechs reproduzierbaren Eingangssignale stehen gemeinsam in
[fftw_comparison_signals.py](../benchmarks/fftw_comparison_signals.py). Eine
nachträgliche Varianznormierung jeder Zufallsrealisierung findet nicht statt.

| Signal | Festgelegtes Modell |
|---|---|
| Weißes Rauschen | Einseitige Quell-NSD 10 nV/√Hz |
| Rosa Rauschen | Quell-PSD \((10\,\mathrm{nV}/\sqrt{\mathrm{Hz}})^2\,0{,}1\,\mathrm{Hz}/f\), endliche Fourier-Synthese, DC null |
| Braunes Spektrum | Quell-PSD \((10\,\mathrm{nV}/\sqrt{\mathrm{Hz}})^2(0{,}01\,\mathrm{Hz}/f)^2\), endliche Fourier-Synthese, DC null |
| Rauschmischung und Linien | Summe der drei unabhängigen Rauschanteile; Sinuslinien bei 0,12345, 3,123456 und 12,34567 Hz mit 30, 50 und 20 nV RMS |
| Sinus zwischen FFT-Bins | 1 µV RMS, vollständiger FFT-Bin-Index mit Bruchteil 0,37 |
| Große Gleichspannung | 10 V plus weißes Rauschen mit 1 nV Zeitbereichs-RMS, Eingang in Float64 |

Die Quellkurven sind Modelle des Eingangssignals, keine exakten Erwartungswerte
jedes endlichen, gefensterten und detrendeten Schätzers. Bei der Rauschmischung
zeigt die Quellkurve ausschließlich den Rauschanteil.

Die Prüfung trennt mehrere Aussagen:

- **Implementierungsgleichheit:** Jeder optimierte FFTW-Pfad wird gegen seine
  eigene bisherige Implementierung geprüft. Aufbewahrt werden absolute
  Abweichungen, relative Abweichungen an nicht verschwindenden Referenzwerten,
  Abweichungen relativ zum Referenzmaximum und die Zahl veränderter Punkte.
  Ein absoluter Grenzwert nahe spektralen Nullen wird zusammen mit dem
  relativen Grenzwert angegeben; er ersetzt nicht die rohen Fehlerwerte.
- **Rauschstreuung:** Die Streuung von
  \(\mathrm{NSD}/\sqrt{S_\mathrm{Quelle}}\) über Frequenzpunkte ist eine
  beschreibende Statistik korrelierter Punkte. Eine unabhängige
  Monte-Carlo-Aussage erfordert getrennte Zufallsrealisierungen. Der vollständig
  FFTW-berechnete Bereich muss zusätzlich zur Gesamtstatistik ausgewiesen
  werden; bei kürzeren Datensätzen kann eine feste Untergrenze von 0,05 Hz
  noch innerhalb der LPSD-Teilberechnung liegen.
- **Linienbreite:** PSD-FWHM wird bei halber PSD-Peakhöhe bestimmt, NSD-FWHM bei
  halber NSD-Peakhöhe, also einem Viertel der PSD-Peakhöhe. Schnittpunkte aus dem
  vorhandenen Raster sind Interpolationsschätzungen mit dessen begrenzter
  Auflösung. Ein separat berechneter dichter Filterantwortscan ist als
  Filterantwort zu kennzeichnen und erzeugt keine zusätzlichen geschätzten
  Spektrumpunkte.
- **Leckboden:** Der isolierte Sinus wird sowohl im Peakbereich als auch fern
  vom Peak in absoluten PSD/NSD-Einheiten gezeigt. Eine gute Übereinstimmung
  der Hauptkeulenbreite rechtfertigt kein Ausblenden kleiner, unterschiedlicher
  Nebenkeulen oder numerischer Böden.

Die zugehörigen Prüfroutinen liegen in
[fftw_comparison_metrics.py](../benchmarks/fftw_comparison_metrics.py).
Die Regressionstests für
[gemeinsame FFTW-Operationen](../test_fast/test_fftw_pipeline_ops.py),
[vorbereitete LPSD-Teilmengen](../test_fast/test_prepared_lpsd.py) und den
[Hybrid](../test_fast/test_matched_smoothing.py) behandeln gezielt diese
Implementierungsgrenzen. Den Abschlussstatus des vollständigen Testlaufs
enthält der Abschnitt „Regressionsstatus“; die Rohprüfwerte der Spektren
bleiben in den gesonderten Messdateien nachvollziehbar.

## Speichergrenzen und verbleibende Einschränkungen

Die Gewichts- und Koeffizientencaches besitzen getrennte Obergrenzen. Metadaten
geben neben den konfigurierten Limits die tatsächlich verwendeten Bytes sowie
gespeicherte und nicht gespeicherte Frequenzen an. `max_working_mb` begrenzt
gesondert gleichzeitige temporäre Reservierungen der LPSD-Vorbereitung. Ein
einzelner Punkt, dessen Bedarf größer ist, wird allein bearbeitet; dieses
Arbeitsbudget ist deshalb kein harter Prozessspeichergrenzwert.

Optional begrenzt `total_cache_mb` die Nutzdaten beider dauerhaften Caches
gemeinsam. Zuerst wird das Gewichtslimit auf diese Obergrenze begrenzt; von
diesem gemeinsamen Budget werden anschließend nur die tatsächlich angelegten
Gewichtsbytes abgezogen. Der LPSD-Koeffizientencache erhält höchstens den Rest.
`window_cache_mb=None` gibt ihm das vollständige Restbudget und verlangt ein
explizites Gesamtlimit. Ein numerisch angegebenes LPSD-Limit, auch null, bleibt
zusätzlich wirksam. Ohne `total_cache_mb` bleiben beide Grenzen unabhängig.
Der Vergleichslauf bildet diese Auswahl mit `--total-cache-mb` und
`--auto-low-cache` ab; `--smoothing-workers` steuert gesondert die
Glättungsarbeiter und `--bluestein N:M` eine ausdrücklich ausgewählte
DFT-/Faltungslänge.

Zum tatsächlichen Speicherbedarf kommen unter anderem Eingangsdaten, FFT-Puffer,
Fenster, Leistungsspektrum, Ausgaben, Python-Objekte, temporäre Vorbereitungen
und FFTW-interner Planspeicher hinzu. Auch der bekannte Bluestein-Pufferbedarf
ist keine RSS-Obergrenze. Mehrere gleichzeitig gehaltene vorbereitete Objekte
vervielfachen ihren dauerhaften Bedarf. Das gewählte Cachebudget von 2 GiB
garantiert folglich weder 2 GiB RSS noch die Einhaltung eines
Gesamtmaschinenlimits. Der gesamte Hauptmessprozess erreichte einen
aufgezeichneten RSS-Höchstwert von 7,8875 GiB. Darin waren zeitweise alte und
neue Vergleichsobjekte zugleich resident; dieser Wert ist kein isoliertes
Speichermaß einer einzelnen optimierten Pipeline.

Der bekannte Defekt der ursprünglichen LPSD-Mittelwertrekurrenz bleibt
unverändert: Bei mindestens zwei Segmenten verwirft die Aktualisierung mit
Divisor `ii` statt `ii + 1` in exakter Arithmetik den Beitrag des ersten
Segments. Auch die ursprüngliche zweite Momentenaktualisierung ist keine
validierte laufende Stichprobenvarianz. Die vorbereitete Teilberechnung erhält
dieses Referenzverhalten; `psd_std` wird nicht als unabhängige
Unsicherheitsschätzung verwendet. Details stehen unter
[Inherited statistical defects](numerics.md#inherited-statistical-defects).

SIMD-Reduktionen, native Formelauswertung, der optionale Float32-Gewichtscache
und Bluestein können jeweils die Rundung verändern. CPU, Compiler,
Mathematikbibliothek und Gleitkommaformat bleiben relevante Einflüsse.
Vollständige Vorher/Nachher-Prüfungen mit den tatsächlichen finalen Optionen
sind daher Teil des Ergebnisses, auch wenn Raster und mathematischer Schätzer
unverändert bleiben.

## Berichte aus gespeicherten Messdaten erzeugen

Der Renderer liest Messdaten, berechnet keine neuen Benchmarkzeiten und erzeugt
PSD-, NSD-, Laufzeit-, Peak-/Leckboden- sowie Vorher/Nachher-Diagramme und einen
statischen HTML-Bericht:

```bash
python benchmarks/render_fftw_optimized.py \
  --results /pfad/zum/abgeschlossenen_lauf.json \
  --output-dir /pfad/zu/diagrammen
```

Die zugehörige Spektrendatei wird standardmäßig unter demselben Basisnamen mit
der Endung `.spectra.npz` erwartet; `--spectra` kann ihren Pfad ausdrücklich
angeben. PNG und SVG enthalten ausschließlich aufgezeichnete Kurven und Zeiten.
Fehlende Messkategorien werden nicht durch geschätzte Zeiten ersetzt.

### Zusätzliche Studie zu größeren Caches

Der getrennte [Tuner](../benchmarks/bench_fftw_tuning.py) kann die gemeinsamen
Cachegrenzen von 2 und 4 GiB bei 10 000 000 beziehungsweise 30 000 000 Proben
vergleichen. Die abgeschlossene ergänzende Untersuchung umfasst je Grenze
zwei Blöcke mit je drei vollständigen NSD-Aufrufen; die FFTW-Threadzahlen
bleiben vier beziehungsweise acht. Jeweils nur eine große vorbereitete
Konfiguration wird im Speicher gehalten. Fenster, Schätzer, Gewichtstyp und
Eingangssignal bleiben unverändert.

Die Konfigurationen laufen in getrennten seriellen Blöcken. Diese Studie ist
kein gepaarter Vergleich gegen die ursprüngliche Implementierung und ersetzt
nicht die Hauptmessung mit 2 GiB. Ihr Gesamtmedian wird aus allen einzelnen
Rohzeiten einer Konfiguration berechnet. Zusätzlich werden die separaten
Blockmediane, Minimum/Maximum und die Anzahl der Aufrufe angegeben, damit
Unterschiede zwischen Blöcken erkennbar bleiben.

Die tatsächlichen Gewichtsbytes, gespeicherten LPSD-Koeffizientenbytes und
Anzahlen gespeicherter q-Frequenzen werden aus den Metadaten übernommen.
Ein eingestelltes Limit von 4 GiB bedeutet nicht, dass der Cache genau 4 GiB
belegt. Der Prozess-RSS-Höchstwert gehört zur gesamten Studienausführung;
bei mehreren Konfigurationen im selben Prozess lässt er sich nicht einer
einzelnen Cachegrenze zurechnen.

### Gemessene Cache-Studie

Die folgenden Gesamtmediane werden aus jeweils sechs Rohzeiten gebildet.
Die beiden Blockmediane bleiben getrennt angegeben. Diese Beobachtungen
ersetzen keine gepaarte Hauptmessung gegen die ursprüngliche Implementierung.

| N | Limit [GiB] | Gesamtmedian NSD [ms] | Block 1 / 2 [ms] | Minimum–Maximum [ms] |
|---:|---:|---:|---:|---:|
| 10 000 000 | 2 | 494,178 | 530,704 / 465,595 | 415,183–559,183 |
| 10 000 000 | 4 | 400,986 | 403,145 / 398,828 | 333,784–433,906 |
| 30 000 000 | 2 | 2126,238 | 2065,770 / 2186,706 | 1912,329–2735,101 |
| 30 000 000 | 4 | 1900,966 | 1915,297 / 1868,622 | 1551,627–2590,381 |

Das Verhältnis der Gesamtmediane beträgt 1,232 bei 10 Millionen und
1,119 bei 30 Millionen Proben zugunsten des größeren Limits. Beide
Blockmediane liegen mit 4 GiB niedriger, die beobachteten Spannweiten
überlappen jedoch. Die Studie läuft seriell auf demselben geteilten Host;
sie misst einen begrenzten zusätzlichen Speicherhebel und keine allgemeine
Hardwarekonstante. Das festgelegte Hauptprofil bleibt bei 2 GiB.

| N | Limit [GiB] | Gewichte [Bytes] | LPSD-Koeffizienten [Bytes] | Beide Caches [Bytes] | Gespeicherte q-Frequenzen |
|---:|---:|---:|---:|---:|---:|
| 10 000 000 | 2 | 336 323 188 | 1 810 359 552 | 2 146 682 740 | 25 / 83 |
| 10 000 000 | 4 | 336 323 188 | 2 789 844 816 | 3 126 168 004 | 83 / 83 |
| 30 000 000 | 2 | 1 004 027 496 | 1 131 347 552 | 2 135 375 048 | 3 / 78 |
| 30 000 000 | 4 | 1 004 027 496 | 3 287 327 504 | 4 291 355 000 | 10 / 78 |

Bei 10 Millionen Proben passen mit 4 GiB sämtliche 83 q-Frequenzen in den
Cache; die tatsächliche Nutzdatenbelegung beträgt nur 3 126 168 004 Bytes.
Bei 30 Millionen sind zehn statt drei von 78 q-Frequenzen gespeichert.
Die RSS-Höchstwerte der vollständigen Studienprozesse betragen 3,6942 GiB
für N = 10 Millionen und 7,7055 GiB für N = 30 Millionen. Diese Prozessmaxima
können nicht jeweils dem 2- oder 4-GiB-Punkt zugeordnet werden. Die abschließende
Prüfung dokumentiert für beide Studien unveränderte Eingangsdaten.

Datenquellen sind `cache_limit_10000000_final.json` und
`cache_limit_30000000_final.json`, jeweils die Einträge `configurations[]`.

Abgeschlossene Tuner-Dateien im Format `configurations[]` können zusätzlich
an den Renderer übergeben werden:

```bash
python benchmarks/render_fftw_optimized.py \
  --results /pfad/results_final.json \
  --prime-quality /pfad/quality_prime_final.json \
  --cache-study /pfad/cache_limit_10000000_final.json \
                /pfad/cache_limit_30000000_final.json \
  --output-dir /pfad/zu/diagrammen
```

Dadurch entstehen zusätzlich `cache_budget_nsd.png` und
`cache_budget_nsd.svg` sowie ein eigener Abschnitt mit Tabellen und eingebetteten
Rohdaten im HTML-Bericht. Das Haupt-JSON wird dabei nicht verändert. Die
Dateien müssen mit der abschließenden Eingangsprüfung des Tuners vollständig
geschrieben sein; Zwischenstände werden nicht als fertige Cache-Studie
ausgegeben.

### Ablage und Herkunft der Ergebnisdateien

Die Rohdaten und erzeugten PNG/SVG/HTML-Dateien sind kein Bestandteil des
Git-Quellbaums. Sie werden im getrennten Ergebnisordner beziehungsweise
ZIP `fftw_optimized_results_2026-10-10` geliefert. Der Hauptlauf besteht aus
`results_final.json` und `results_final.spectra.npz`; die zusätzliche
Primlängenprüfung aus `quality_prime_final.json` und ihrer NPZ-Datei.
Die beiden Cachedateien behalten ihre eigenen seriellen Tuningprotokolle.

`--prime-quality` ergänzt im HTML eine eigenständige Fehlertabelle samt
Hinweis auf sehr kleine Leckwerte; es verändert weder Hauptspektren noch
Laufzeiten. Der erzeugte Bericht heißt `report_fftw_optimized.html` und
bettet alle Diagramme als SVG sowie die verwendeten JSON-Protokolle ein.
Die Dateien können anhand der dokumentierten Quelltext- und Bibliothekshashes
dem gemessenen Arbeitsstand zugeordnet werden. Reproduktionspfade wie
`/pfad/results_final.json` sind vom jeweiligen lokalen Ergebnisordner abhängig.
