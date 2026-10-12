# LPSD/LNSD und FFTW: Speicher, Punktzahl und große Datensätze

Die gerettete Hauptmatrix ist mit **132 von 132 vollständig abgeschlossenen
Fällen** verfügbar, einschließlich einer Milliarde Eingangswerte und einer
tatsächlichen Verdopplung der Ausgabepunkte. Weitere vollständige
Optimierungsversuche belegen eine kleinere native Speicherbelegung und
Vorteile durch zusätzliche Glättungsworker. Die ungewöhnlichen nativen
1B-Laufzeiten bleiben als Einzelbeobachtungen erkennbar.

Die [kompakte Messzusammenfassung](../benchmarks/results/recovered-campaign-20261012.json)
enthält die Zahlen, genaue Timergrenzen, historische Quellstände und
SHA256-Identitäten der ursprünglichen Ergebnisdateien. Sie enthält keine
Rohsignale, vollständigen Spektren oder persönlichen Rechnerdaten. Die
folgenden historischen Messungen verwenden **E/G, `1.0.6+fast.7`**; die
Weiterentwicklung `1.0.6+fast.8` wird dadurch nicht nachträglich als gemessen
ausgegeben. Teilnahme und Steueroptionen beschreibt die
[Benchmarkkampagne](benchmark-campaign-2026-10.md).

## Umfang und Vergleichsmaßstab

E bezeichnet Commit `7d0f52b9096cbe2f74064d86f2fe6b9fa69320c9`. Die
Hauptmatrix lief auf Linux x86-64 mit acht CPU-Äquivalenten, einer harten
8-GiB-cgroup-Grenze, 6,25 GiB Arbeitsbudget, 24 GiB temporärem Dateibudget
und System-FFTW 3.3.10. Sie verwendet acht FFTW-/native Worker und vier
Glättungsworker. Sie ist kein Vergleich mehrerer physischer Rechner und
**kein tatsächlich ausgeführter 32-GB-RAM-Versuch**.

100.000, 1.000.000 und 1.000.003 Eingangswerte wurden jeweils mit sechs
synthetischen Signalen geprüft: weißes, rosa und braunes Rauschen,
Rauschen mit Tönen, isolierter Off-bin-Ton sowie DC mit Nanovolt-Rauschen.
Die vier größeren Längen verwenden weißes Rauschen. Jede Kombination aus
Methode, Signal, N und Punktraster hat in der Hauptmatrix genau einen
vollständigen Aufruf; die getrennten Gegenpaare werden nicht dazugemischt.

| Methode | Berechneter Schätzer |
|---|---|
| Native LPSD/LNSD | Ursprünglicher frequenzabhängiger LPSD-Plan, native Projektionen und begrenzter Arbeitsbereich |
| Einfaches FFTW | Vollständiges Periodogramm mit logarithmischer Aggregation; stärker schwankender Schätzer |
| FFTW-Hybrid | FFTW, frequenzabhängige Glättung und native tiefe Frequenzen zur Annäherung an LNSD-Glätte und Hauptkeulenbreite |

PSD und NSD entstehen **gemeinsam aus einem Spektrum**. `compute_wall_s`
misst einen frischen vollständigen Aufruf mit Planung, Arbeitsarrays,
Ausgaben und Freigabe. `compute_cpu_s` summiert dessen Prozess-CPU über alle
Threads. Eingabevorbereitung einschließlich erster Prüfsumme steht in
`input_setup_s`; die gesamte Kindprozesslaufzeit einschließlich Importen und
Prüfungen in `process_wall_s`. Es gibt keinen eigenen Importtimer und keinen
separaten Timer für jede Prüfsumme. Parallel überlappende Punkt-Wallzeiten
sind keine addierbaren exklusiven Phasen oder CPU-Zeiten.

## Echte Verdopplung der Punkte

Der Plannerparameter J ist nicht die Zahl der ausgegebenen Frequenzen.
`--point-factors 1 2` löst deshalb J für beide tatsächlichen Ausgabezahlen
separat auf. Ein bloßer Wechsel von J=1000 auf J=2000 reicht dafür nicht.

| Eingangswerte N | J Basis | Punkte Basis | J für doppelte Ausgabe | Punkte doppelt |
|---:|---:|---:|---:|---:|
| 100.000 | 1000 | 469 | 2425 | 938 |
| 1.000.000 | 1000 | 577 | 2240 | 1154 |
| 1.000.003 | 1000 | 577 | 2240 | 1154 |
| 10.000.000 | 1000 | 651 | 2169 | 1302 |
| 30.000.000 | 1000 | 678 | 2147 | 1356 |
| 100.000.000 | 1000 | 704 | 2130 | 1408 |
| 1.000.000.000 | 1000 | 744 | 2105 | 1488 |

Das dichtere Raster verändert Segmentlängen, Bandbreiten und Mittelungen.
Es interpoliert keine vorhandene Kurve. Schmalere Tonspitzen und größere
Rauschwelligkeit sind dadurch gemeinsam zu erwarten.

## Vollständige Wallzeiten der Hauptmatrix

Alle Tabellenzeilen hier verwenden dieselbe kanonische Weißrauschdefinition.
Zeitangaben sind Sekunden für **PSD und NSD zusammen**.

### Basisraster

| N | Punkte | Speicherpfad | Einfaches FFTW [s] | Hybrid [s] | Native LPSD/LNSD [s] |
|---:|---:|---|---:|---:|---:|
| 100.000 | 469 | RAM | 0,015575 | 0,047732 | 0,083615 |
| 1.000.000 | 577 | RAM | 0,038133 | 0,134962 | 0,146196 |
| 1.000.003 | 577 | RAM | 0,213587 | 0,313626 | 0,149753 |
| 10.000.000 | 651 | RAM | 0,231309 | 1,293483 | 1,368361 |
| 30.000.000 | 678 | RAM | 0,730747 | 3,535277 | 4,822268 |
| 100.000.000 | 704 | RAM | 1,940388 | 13,138587 | 19,945835 |
| 1.000.000.000 | 744 | Dateimapping | 1068,033134 | 1154,043201 | 5350,291715 † |

### Doppelte Ausgabezahl

| N | Punkte | Speicherpfad | Einfaches FFTW [s] | Hybrid [s] | Native LPSD/LNSD [s] |
|---:|---:|---|---:|---:|---:|
| 100.000 | 938 | RAM | 0,017848 | 0,071882 | 0,125918 |
| 1.000.000 | 1154 | RAM | 0,036659 | 0,204256 | 0,327104 |
| 1.000.003 | 1154 | RAM | 0,212821 | 0,440523 | 0,315696 |
| 10.000.000 | 1302 | RAM | 0,237039 | 1,707763 | 2,919226 |
| 30.000.000 | 1356 | RAM | 0,662347 | 4,726737 | 10,106378 |
| 100.000.000 | 1408 | RAM | 1,980356 | 18,513623 | 37,049023 |
| 1.000.000.000 | 1488 | Dateimapping | 1090,526457 | 1165,041506 | 892,382996 † |

† Die native 1B-Anomalie ist **kein belegter Vorteil durch mehr Punkte**.
Der dichtere Plan leistet mehr direkte Projektionsarbeit und streamt 190
statt 90 Punkte. Beide E-Aufrufe behalten ihren anfänglichen
1.431.655.760-Byte-Pool bis zum Ende. Die vorhandenen Daten identifizieren
die Ursache des großen Zeitunterschieds nicht eindeutig.

### Prozess-CPU der großen Fälle

| N / Punkte | Einfaches FFTW [CPU-s] | Hybrid [CPU-s] | Native [CPU-s] |
|---|---:|---:|---:|
| 100M / 704 | 4,258677 | 57,521027 | 143,149044 |
| 100M / 1408 | 4,006295 | 84,173953 | 259,720497 |
| 1B / 744 | 313,014930 | 820,434469 | 6363,568841 |
| 1B / 1488 | 317,055168 | 1155,314722 | 4474,825153 |

CPU/Wall ist ein mittlerer Ressourcenverbrauch, keine gemessene Zahl aktiver
Worker. Beim 1B-Hybrid steigt die CPU-Zeit mit doppeltem Raster um 40,82%,
obwohl die beobachtete Wallzeit nur um 0,95% wächst: gegenläufige FFT- und
Low-Phasen verdecken einen Teil der zusätzlichen Arbeit.

## Qualität: Rauschwelligkeit und Tonbreite

Die folgenden deskriptiven Standardabweichungen beziehen sich auf
NSD/(10 nV/√Hz) im Band 0,01–20 Hz. Sie sind in Prozent angegeben.

| Methode | 100M / 704 | 100M / 1408 | 1B / 744 | 1B / 1488 |
|---|---:|---:|---:|---:|
| Einfaches FFTW | 1,900484% | 2,898237% | 0,556430% | 0,815369% |
| FFTW-Hybrid | 0,383816% | 0,590826% | 0,154570% | 0,226894% |
| Native LNSD | 0,408696% | 0,601141% | 0,155739% | 0,226388% |

Die Rasterpunkte sind korreliert; diese Werte sind keine Konfidenzintervalle
aus unabhängigen Versuchen. Der punktweise relative NSD-RMS-Unterschied
Hybrid/LNSD liegt bei 1B bei 0,034928% beziehungsweise 0,061719%; die
Maximalabweichungen liegen bei 0,172676% beziehungsweise 0,375045%.
Das bewertet diese Weißrauschrealisierung, keine beliebigen Eingangssignale.

Beim isolierten 3,1230185-Hz-Ton mit N=1M entstehen folgende interpolierte
Halbwertsbreiten:

| Methode | PSD-FWHM / 577 | NSD-FWHM / 577 | PSD-FWHM / 1154 | NSD-FWHM / 1154 |
|---|---:|---:|---:|---:|
| Hybrid | 0,115708 Hz | 0,159071 Hz | 0,051658 Hz | 0,071613 Hz |
| Native | 0,115708 Hz | 0,159071 Hz | 0,051658 Hz | 0,071613 Hz |

PSD-Halbmaximum bedeutet halbe Leistung, NSD-Halbmaximum halbe Amplitude.
Nur wenige gespeicherte Punkte liegen in diesen Hauptkeulen. Die
Nachkommastellen beschreiben die Kreuzungsinterpolation und keine ebenso
feine physische Frequenzauflösung. Ähnliche Hauptkeulen bedeuten auch keine
Gleichheit aller sehr kleinen Nebenkeulen.

## Gesonderter E/G-Nachtest: native Poolfreigabe

G (`29ea18567be652ca289e8bf61e3ce5587663ccec`) verändert den nativen
Scratch-Lebenszyklus bei geeigneten Dateieingaben. Nach vollständig
abgearbeitetem Präfix wird der große eigene Pool freigegeben und durch den
für den restlichen Plan nötigen kleineren Pool ersetzt. C-Binärdatei,
Input, Plan, Streamingmaske und numerische Rechenwege bleiben gleich.

| Frisches 1B-/744-Punkte-Paar | E | G |
|---|---:|---:|
| Vollständige Wallzeit | 5233,983317 s | 1570,825963 s |
| Vollständige Prozess-CPU | 6238,077622 s | 3012,479671 s |
| Prozessbezogene Geräte-Reads | 416,818 GB | 202,570 GB |
| Major Faults | 89.624.946 | 19.408.390 |
| User-CPU | 1558,164889 s | 1672,783486 s |
| System-CPU | 4679,912796 s | 1339,696270 s |

Beide Kinder wurden regulär beendet und eingesammelt. Sämtliche **744
Frequenz-, PSD- und NSD-Werte sind exakt gleich**, auch zur früheren
E-Basismessung. Bei G sinkt die Poolkapazität nach j=278 von
1.431.655.760 auf 132.638.128 Bytes. RssAnon liegt gegen Ende bei ungefähr
236 MB statt E 1,528 GB; der Anfangspeak bleibt unverändert.

Die beobachtete vollständige Zeitverkürzung beträgt 69,99%, entsprechend
E/G=3,332. **Dieser ganze Faktor ist nicht kausal der Poolfreigabe
zuzuschreiben:** G ist schon vor der Freigabe deutlich weiter als E. Es gibt
ein geordnetes E→G-Paar, keinen neuen 1B-Gegenlauf und keinen kontrolliert
gleichen Dateicachezustand. Die kleineren Reads und die deutlich gesunkene
System-CPU belegen den unterschiedlichen Pagingverlauf, keine eindeutige
Zerlegung in reine I/O- und Rechenwartezeit.

Bei 100M lauten die vollständigen Zeiten in Reihenfolge E/G/G/E
24,956785 /21,740765 /18,368494 /17,939083 s. Die Zeitrichtung wechselt im
Gegenpaar; Reads und Major Faults sind dort jeweils null. Diese Fälle
bestätigen die Freigabe und Bitgleichheit, keinen konsistenten 100M-Speedup.

## Mehr Glättungsworker bei gleicher Rechenaufgabe

Der separate 4/8-Worker-Screen hält E, FFTW 3.3.10, acht FFTW-/native
Worker, In-place-FFT, vollständiges Fenster und 2048 MiB natives Low-Budget
fest. Zwei wechselnd angeordnete Timingpaare ergeben je Median:

| N / Punkte | Vier Worker | Acht Worker | Änderung der Wallzeit |
|---|---:|---:|---:|
| 30M / 678 | 4,087632 s | 3,421531 s | −16,30% |
| 30M / 1356 | 6,110563 s | 5,151582 s | −15,69% |
| 100M / 704 | 13,158833 s | 10,765189 s | −18,19% |
| 100M / 1408 | 19,610967 s | 16,817353 s | −14,25% |

Der erste 30M-Basisvergleich streut deutlich. Ein separat ausgewiesener
gezielter Gegencheck bestätigt −14,70% Wallzeit bei +5,78% CPU-Zeit. Alle
15 Zeit-/Profilpaare, insgesamt 30 Kinder, liefern bitgleiche Ausgaben.
Die fünf zusätzlichen Profilpaare bestätigen eine kürzere Glättungsphase,
werden aber nicht in die Timingmediane eingerechnet.

`--smoothing-workers 8` macht diese Wahl zugänglich; der zugelassene
Threadcount begrenzt sie. Der Standard bleibt min(4,T). Die portable
Ressourcenprüfung berechnet 9 MiB zusätzliche Reserve je Worker oberhalb
dieses Standards und kann dadurch das sonstige verfügbare Budget ändern.
Sie ist deshalb nicht automatisch dieselbe Konfiguration wie dieser feste
2048-MiB-Screen. Die alten schmalen Glättungs-/Primzahlsupervisoren hatten
einen Fehler bei der PID-Namespace-Zuordnung; ihre RSS-Werte werden nicht
als wirksamer zusätzlicher Guard ausgegeben. Die Kind-Selbstmessungen und
die harte cgroup-Grenze sind getrennt dokumentiert.

## FFT-Varianten unter der tatsächlich erlaubten Genauigkeit

Der Auftrag erlaubt numerisch unbedeutende relative Abweichungen **unter
1%** sowie winzige absolute Unterschiede. Einige frühere Kandidatentests
hatten stattdessen 3 ppm und `atol=0` für jeden ausgegebenen Wert festgelegt.
Die alten Dateien behalten ihren damaligen `failed`-Status; die folgende
Neubewertung ist ausdrücklich eine zusätzliche Beurteilung derselben
gespeicherten Spektren nach der Nutzeranforderung.

### Bluestein bei N=1.000.003

56 Kinder sind regulär abgeschlossen: 48 Timingaufrufe und acht getrennte
Profile. Für Weißrauschen im Basisraster sinkt der vollständige Median
beim einfachen FFTW-R2C-Vergleich von 218,224 auf 123,357 ms; beim Hybrid
von 325,038 auf 262,702 ms. Das entspricht etwa 1,77× beziehungsweise 1,24×.
Originaldatenlänge und Ausgaberaster bleiben erhalten.

Die größten gespeicherten Tonabweichungen betragen **0,101458% PSD und
0,050739% NSD**. Weißrauschen und DC plus Nanovolt-Rauschen sind zwischen
den Routen exakt gleich. Die maximale absolute Tonabweichung über die
geprüften Fälle beträgt 1,505×10⁻³⁶ V²/Hz beziehungsweise
2,118×10⁻²² V/√Hz. Die Tonfälle verfehlten teilweise das alte 3-ppm-Gate,
erfüllen aber den erlaubten 1%-Spielraum. Der zusätzliche persistente
FFT-Pufferbedarf beträgt laut gespeichertem Modell 64.384.048 Bytes;
andere Längen, FFTW-Builds und Rechner benötigen eine eigene Bewertung.

### Packed Real für den Dateipfad

Der ursprüngliche externe Prototyp besteht 26 kleine Verträge mit 562
Checks. Alle 48 festgelegten Hauptvergleiche und 96 vollständigen Aufrufe
wurden ausgeführt; 44 Vergleiche bestehen das alte 3-ppm-Gate, vier reine
Off-bin-Tonfälle im einfachen FFTW-Pfad verfehlen es. Der Hybrid besteht
alle 24 seiner strengen Vergleiche.

Die zusätzliche Auswertung aller **37.656 PSD- und 37.656 NSD-Werte** findet
maximal **0,151066% PSD und 0,075562% NSD**, somit keine 1%-Verletzung.
Es gibt keine nichtendlichen Werte oder ungleich null gewordenen
Nullreferenzen. Die größten absoluten Unterschiede über alle Fälle sind
4,815×10⁻³⁵ V²/Hz und 1,694×10⁻²¹ V/√Hz. Input, Frequenzen, vollständige
Planarrays, Fenster und Vorbereitung stimmen weiterhin exakt überein.

Diese gespeicherte Numerik liefert **keine Performancezahl**. Der alte
Prototyp behielt außerdem die gesamte 16N-Byte-Workspace-Datei und
24N-Dateibelegung bei. Eine halbierte FFT-Länge bedeutet daher weder eine
gemessene halbierte Gesamtzeit noch halbierte gesamte Dateibelegung.
Die neue explizite Steuerung `--disk-fft-algorithm packed-real` besitzt ihre
eigene Integrationsprüfung; sie wird nicht als nachträglich gemessener alter
Prototyp dargestellt.

## Speicher: 32 GB sind hier ein Modell

Für N=1B Weißrauschen ergibt das konservative Zulassungsmodell mit
empirischer 8N-FFTW-Planerreserve und 256 MiB Grundreserve:

| RAM-Konfiguration | Einfaches FFTW | Hybrid |
|---|---:|---:|
| Vollständiges Fenster, getrennte FFT-Puffer | 40,268435456 GB | 44,268435456 GB |
| Vollständiges Fenster, In-place | 32,268435456 GB | 36,268435456 GB |
| Begrenztes Fenster, In-place | 24,270532608 GB | 28,276824064 GB |

GB bezeichnet 10⁹ Bytes; GiB bezeichnet 2³⁰ Bytes. Selbst mit idealisiert
vollständig verfügbarem physischem Speicher bleiben bei 32 dezimalen GB
nach 2 GiB OS-Reserve und MiB-Abrundung höchstens 29,851910144 GB
Arbeitsbudget; bei 32 GiB sind es höchstens 30 GiB. Tatsächliche Belegung
kann das weiter senken. Die Tabellenwerte sind **Zulassungsschätzungen,
keine garantierten RSS-Spitzen und keine Laufzeitmessungen**.

Der begrenzte Fenstermodus spart einen separaten 8N-Byte-Vektor und
regeneriert Koeffizienten in höchstens 262.144 Samples großen Blöcken.
Bei einem einzelnen vollständigen Aufruf und N=1B sind das rechnerisch
8 dezimale GB weniger Arraybedarf. Normierung und ursprüngliche Samples
bleiben erhalten. Die Automatik behält das vollständige Fenster, solange
es mit den übrigen Arrays passt; wiederverwendbare Schätzer behalten ihren
vollständigen Fensterstandard.

Die neue Option `--native-working-mib 384` senkt auf Wunsch ausschließlich
das vorhandene native Arbeitsbudget. 233 kleine G384-v2-Fälle mit 685
Streamingvergleichen und zwei erwarteten Abweisungen sind bestanden.
Ein vollständiger G384-1B-Zeitvergleich liegt in dieser historischen
Kohorte nicht vor. Ein kleinerer Cap kann mehr Cache-Platz lassen, aber
auch zusätzliche Koeffizientenregeneration und weniger Parallelität
verursachen; er wird deshalb nicht pauschal zum schnelleren Standard.

## Dateispaltenstrategie: vollständiges 1B-Paar mit Schreibverstärkung

Der separate 1B-Screen ist am 12. Oktober vollständig abgeschlossen.
Auto-Spaltentransposition und Stride verwenden dieselben Faktoren
[10000,100000], acht FFTW-Threads, 128 MiB FFT-Arbeitsbudget und dieselbe
24N-Byte-Dateibelegung. Das Raster enthält 744 Punkte bei J=1000.

| Vollständiger Aufruf | Auto / Transposition | Stride, gleiche Faktoren |
|---|---:|---:|
| Wallzeit | 1094,372825 s | 847,728836 s |
| CPU-Zeit | 319,013775 s | 243,860602 s |
| Prozess-`read_bytes` | 128.062.980.096 | 111.314.481.152 |
| Prozess-`write_bytes` | 126.082.617.344 | 444.979.793.920 |

Das ergibt **22,537% weniger Wallzeit (1,291×)** und 23,558% weniger
CPU-Zeit in diesem Paar. Frequenzen, PSD und NSD sind exakt gleich;
Input-Hashes und Quellmanifest bleiben unverändert. Beide Kindprozesse
enden erfolgreich, werden regulär eingesammelt, und das eigene temporäre
Verzeichnis ist bereinigt.

Der Zeitvorteil hat einen erheblichen Gegeneffekt: **3,529× so viele
Schreibbytes**. Die beiden vollständigen Spaltentranspositionen des
Auto-Pfads benötigen zusammen 729,403 s. Bei Stride entfallen sie,
während Column-Gather und -Scatter von zusammen 22,084 auf 542,400 s
steigen. Die übergeordneten Phasen enthalten ihre Unterphasen und dürfen
nicht noch einmal mit ihnen addiert werden.

Die Prozess-I/O-Zähler umfassen Cache- und Writeback-Effekte; sie sind
keine präzise Messung physischer SSD-Schreibvorgänge. Es gibt nur dieses
geordnete Auto→Stride-Paar, kein umgekehrtes Gegenpaar. Cache-Vorgeschichte,
Speichergerät und andere Plattformen bleiben daher offene Einflussgrößen.
Dieser abgeschlossene Befund rechtfertigt **keine pauschale Änderung der
automatischen Spaltenstrategie**.

Weitere Anleitungen: [Portabler Benchmark](portable-benchmark.md),
[Datenschutz und Meldungen](benchmark-privacy.md) und
[früheres FFTW-Profil](performance-fftw-optimized.md).
