# Ergänzende Nachweise zur Benchmarkkampagne vom 12. Oktober 2026

Diese Ergänzung dokumentiert die Wiederherstellung und erneute Auswertung
vorhandener Messdaten sowie neu abgeschlossene, unabhängig geprüfte
Spalten- und Direct-log-R3-Nachweise. Die Methoden, großen Zahlentabellen und früheren
Optimierungsbefunde bleiben im [Speicher- und Punktebericht](performance-memory-points.md).
Der [Plattform-Folgebericht](performance-platform-followup.md) und der
[Plattformvergleich](platform-comparison-20261012.md) behalten ihre eigenen
Quellstände, Messumfänge und Bewertungsmaßstäbe.

## Wiederhergestellte E-Matrix und neue Auswertung

Die Originaldatei `full_matrix_E.json` ist bytegenau wiederhergestellt.
Ihr SHA256 lautet
`1b1f9f72a0f86289fe9d333e4a2d5623ddbbf2c26ed4ab86d9ddd1c3f5737117`.
Sie enthält **132 abgeschlossene Fälle** der E-Kohorte,
Commit `7d0f52b9096cbe2f74064d86f2fe6b9fa69320c9`,
`1.0.6+fast.7`, mit FFTW 3.3.10. Diese Daten sind keine nachträgliche
Messung der späteren fast.8-Implementierung oder der
fast.9-Folgeintegration. Die
[vorhandene kompakte Zusammenfassung](../benchmarks/results/recovered-campaign-20261012.json)
ordnet die historischen Quellen bereits getrennt zu.

Der neu erstellte Renderer hat folgende Bindungen an die erhaltenen Daten
geprüft:

- **112.635 gespeicherte Frequenzpositionen**, über alle 132 Spektrensätze
  gezählt; dies ist keine Zahl unterschiedlicher Frequenzlabels.
- **396 Array-Hashes**: je Fall Frequenzen als float64 sowie PSD und NSD
  als float32, einschließlich des ursprünglichen Datentyppräfixes.
- **132 NSD-Ableitungen** über die ursprüngliche Route
  `sqrt(complex64(PSD)).real`.
- **44 Gruppen** mit identischen Eingabedigests und vollständigen
  Frequenzarrays für die drei Verfahren.
- **66 Basis-/Doppelraster-Paare** mit tatsächlich verdoppelter positiver
  Ausgabezahl und passender Eingabe-, Quellen- und nativer Binärbindung;
  bei den FFTW-Verfahren wird zusätzlich deren FFTW-Bindung verglichen.

Die Eingabedigests stammen aus den ursprünglichen Messungen. Die N großen
Eingangssignale wurden für die Auswertung nicht erneut erzeugt oder gelesen.
Es gab keinen neuen Schätzer- oder Benchmarkaufruf.

Im separaten Ergebnisartefakt liegen jetzt **20 wissenschaftliche
PNG-/SVG-Paare**, sechs CSV-Exporte und ein eigenständiger HTML-Bericht.
Die Abbildungen decken alle sechs Signale bei 100.000, 1.000.000 und
1.000.003 Werten für beide Raster ab; hinzu kommen die großen
Weißrauschfälle, Off-bin-Details, Welligkeit und vollständige
Berechnungszeiten. `analysis_validation.json`, `analysis_summary.json`,
`render_receipt.json` und `visual_qa.json` halten die Prüfungen und
Ableitungen fest. Sechs zentrale PNGs wurden visuell geprüft; der enge
unabhängige Quellenreview fand für genau diese E132-Datei keinen Blocker.

Eine anschließende Integritätsprüfung fand die erste
`recovered_nsd_large_n_points.png` bereits in ihrer Sicherung unvollständig.
Nur diese PNG wurde aus denselben Rohdaten mit einem gepinnten
Rendererfragment neu exportiert, auf PNG-Integrität geprüft und visuell
kontrolliert. Ihr korrigierter SHA256 ist
`12e7fad824cc1f7998e348c703b1c985769fe6ebcddeb5a30c5e1dc3bb6f8dd2`.
Alle 20 SVGs und die 19 übrigen PNGs waren intakt; der vollständige
HTML-Bericht mit 20 eingebetteten SVGs bleibt unverändert. Dies belegen
`png_integrity_observation_20261012.json`,
`single_png_reexport_receipt.json`, `reexport_large_nsd_png.py` und die
aktualisierte `visual_qa.json`. Es wurde kein Schätzer erneut ausgeführt.

`render_recovered_matrix.py` ist **neu geschriebener Auswertungscode**.
Sein SHA256 ist
`ee148f438911918fa3e33d8fef039a3c8be446c1615184577d753e624cbd5b55`.
Damit wird keine Bytegleichheit mit dem verlorenen früheren Renderer oder
Analyzer behauptet. Die erzeugten Spektralarrays und Rohdaten verbleiben
im Ergebnisartefakt; diese Dokumentation übernimmt keine Rohsignale,
Spektralarrays oder personenbezogenen Umgebungsdaten in das Repository.

### Bedeutung der abgeleiteten Kennzahlen

Jede Hauptmatrixzeit ist weiterhin **ein vollständiger frischer Aufruf
für PSD und NSD gemeinsam**. Sie wird nicht halbiert oder als eigener
PSD-/NSD-Timer interpretiert. Die RAM- und dateibasierten Messpfade bleiben
unterscheidbar. Die ungewöhnlichen nativen 1B-Zeiten sind mit einem
Einzelbeobachtungshinweis versehen; ihr Abstand belegt keinen kausalen
Geschwindigkeitsvorteil durch zusätzliche Punkte.

Die Rauschwelligkeit ist die Standardabweichung der gespeicherten NSD,
geteilt durch die feste Quell-NSD, über gleichgewichtete vorhandene
Bandpunkte mit `ddof=1`. Das ist eine beschreibende Größe einer
Realisierung, kein Konfidenzintervall. Die FWHM wird für gespeicherte PSD
und NSD getrennt an deren jeweiliger Halbhöhe bestimmt. Beide Kreuzungen
müssen belegt sein; nur zwischen benachbarten Frequenzlabels wird linear
interpoliert. Fehlende Kreuzungen bleiben als nicht begrenzte Breite
gekennzeichnet. Die NSD-Auswertung verwendet hier direkt die gespeicherte
float32-NSD; die ursprüngliche Hilfsmetrik mit Float64-Wurzel der PSD bleibt
zum Vergleich erhalten. Mehr Punkte verändern zugleich Fensterlängen und
Bandbreiten.

## G/G2 bei 1B und 1488 Punkten: numerische Evidenz, keine Zeitfreigabe

Der erhaltene G→G2-Versuch mit einer Milliarde Eingangswerten und 1488
Ausgabepunkten bleibt als **`failed`** klassifiziert. Er enthält vollständige
Frequenz-, PSD- und NSD-Arrays, die untereinander und zur nativen
Doppelraster-Zeile der E132 bitidentisch sind. **47 unabhängige
Stdlib-Prüfungen** bestätigen die gespeicherte Raster-, Eingabe-,
Mittelungs- und Profilkonsistenz sowie die beiden protokollierten
G2-Poolverkleinerungen. Die erhaltenen Anon-Traces stützen die tatsächlichen
Freigaben.

Der Infrastrukturverlust und der `FileNotFoundError` bei der abschließenden
Quellenprüfung schließen diesen Versuch jedoch von der Laufzeitbewertung
aus. Er liefert **keine zugelassene Runtimequote oder Speedup-Aussage**.
Die numerische Gleichheit und die Lebenszyklusbeobachtungen gelten für diese
Weißrauschrealisierung; sie ersetzen weder eine allgemeine
Genauigkeitsqualifikation noch eine Freigabe als Standardimplementierung.
Der ursprüngliche Fehlerstatus bleibt erhalten.

Die getrennten Begleitbelege heißen
`native_1b_infrastructure_review.md` und
`native_1b_infrastructure_review.json`. Ihre SHA256-Werte sind
`75b5459ee7bbd3b6e15bf09fb055380fbfffea14bdd68b05723a210a3c724d78`
beziehungsweise
`cb18864493a771104c43516f933b4e79d3846fc4c7ae6e9a05dcb9b733383f54`.

## Abgeschlossener FFTW-Spaltenvergleich bei einer Milliarde Werten

Der neue Lauf `paired_disk1b_columns_E_policy_followup.json` hat **70
unabhängige gespeicherte-Daten- und Metadatenprüfungen** bestanden. Die
Reihenfolge ist `strided_default → auto`: zuerst Faktoren
31.250 × 32.000 mit strided-Zugriff, anschließend 10.000 × 100.000 mit zwei
vollständigen Transpositionen. Beide Aufrufe liefern alle 744 Frequenz-,
PSD- und NSD-Werte bitidentisch bei gleicher kanonischer Weißrauscheingabe,
fs=50 Hz, Kaiser PSLL=200 und gebundenen Quellen-/Bibliotheksidentitäten.

Die vollständigen gemeinsamen PSD+NSD-Wall-Zeiten betragen
**784,941853812 → 1276,722081807 s**, die CPU-Zeiten
**265,535434706 → 334,026896960 s**. In diesem einzelnen seriellen Paar hat
`strided_default` 38,519 % weniger Wall-Zeit, aber 41,311 % mehr erfasste
Prozess-Schreibbytes: 210.385.133.568 gegenüber 148.881.145.856 B.
Die Prozess-Lesebytes betragen 102.966.054.912 gegenüber 132.546.170.880 B.
Diese Zähler enthalten Cache-/Writeback-Effekte und die bestehenden
Checkpointkosten; sie sind keine exklusiven SSD-Gesamtschreibmengen.

**Faktorisierung und Spaltenzugriff ändern sich gemeinsam.** Der Lauf ist
kein umgekehrtes Paar bei gleichen Faktoren und begründet keinen
automatischen Standardwechsel. Reihenfolge und Dateicache werden nicht
herausgerechnet. Die vollständigen Timer enthalten die bestehenden
Diagnosen, Callbacks und Cleanup; verschachtelte Unterphasen werden nicht
zusätzlich addiert. Die tatsächliche Controllergrenze ist 7200 s, je Kind
3600 s. Der unveränderte pauschale Rohtext mit 10.800 s wird als
Beschreibungsabweichung dokumentiert. Beide Kinder schließen mit Exit 0,
sind eingesammelt und die temporäre Ablage ist bereinigt.

Die getrennten Belege heißen
`column_policy_followup_review/RESULT_REVIEW.md` und `RESULT_REVIEW.json`.
Der geprüfte Parent-SHA256 ist
`67964f63b22497c8b4bd82a45671635da7eb3e24d6dbe0c87e9368525a37b2dd`.

## Neues Direct-log-R3-Kleingate: vollständige numerische Qualifikation

Der neu gebaute externe R3-Kandidat ist eine eigene Quellenrevision.
**1080 unabhängige Prüfungen seiner gespeicherten Belege bestehen**;
der tatsächliche feste Lauf umfasst 30 kleine Verträge, 24
Hauptkonfigurationen und 48 vollständige Aufrufe. Die 24 Konfigurationen
enthalten alle sechs unveränderten Signale bei N=100.000 und N=1.000.000
auf Basis- und Doppelraster: 469/938 beziehungsweise 577/1154 Punkte.
Die kleinen Verträge enthalten zusätzlich 141 gespeicherte numerische
Vergleiche mit ihren jeweils festgelegten Grenzen.

Je **18.828 gepaarte PSD- und NSD-Werte sind bitidentisch**; maximale
absolute und relative Differenz sind jeweils 0. Die vollständige
Qualifikation bleibt `rtol=3e-6`, `atol=0`, einschließlich Nullen und sehr
kleiner Bins. Der größte vollständige Parseval-Relativfehler beträgt
`1.1693803047441206e-15` gegenüber der Grenze `1e-10` bei absoluter
Toleranz 0. Eingabe, fünf Planarrays, originale Cuts, Fenster, vorbereitete
Samples und vollständige komplexe FFT-Matrix sind innerhalb jedes Paars
gebunden. Die unabhängige vollständige Energie sowie Gruppen-, DC- und
Nyquist-Zähler stimmen; die ursprüngliche NSD-Ableitung ist geprüft.

Parent und tatsächlicher Kindbericht schließen erfolgreich, Exit 0 ist
eingesammelt und die eigene temporäre Ablage entfernt. Die vorhandenen
Timer enthalten absichtliche Hashbeobachter für Fenster, vorbereiteten
Eingang und vollständige FFT-Matrix. **Sie werden nicht für
Geschwindigkeitsfaktoren oder Performance-Mediane verwendet.** Das Gate
belegt diesen endlichen Korpus und den tatsächlich gebauten Helper; es
belegt keine Milliarden-Performance oder allgemeine Plattformfreigabe.

Primärbeleg ist `direct_log_power_recovery_r3/numeric_gate_run1.json`,
SHA256
`3b18225111d31bec582714fa4f8681127beb65979c3bf323d3285869b43be355`,
mit tatsächlichem Kindbericht und Log. Der unabhängige Review liegt als
`direct_log_power_r3_result_review/RESULT_REVIEW.md` und
`RESULT_REVIEW.json` vor. Quellenmanifest:
`82e650a3194bbf5849eb3a26980b04dfabda028e8792f2bdf87ca54c35a6a6ae`;
geprüfter neuer Helper:
`db482ee6863ea4f11b87c33916c87e73a6ed46c652ad0eb7d553985805102734`.
Diese tatsächlich vorhandenen R3-Belege ersetzen keine fehlende R2-Rohdatei.

## R3 bei einer Milliarde Werten: vor dem Direct-Aufruf blockiert

Der abschließend geprüfte `result_E_R3_run1.json` bleibt **`failed`**.
Die tatsächliche Jobfolge ist genau `prepare → E`: Vorbereitung und
einfaches FFTW der E-Referenz wurden abgeschlossen. **Direct-log R3 wurde
nicht gestartet.** Der unabhängige Ergebnisreview bestätigt 40 skalare
Metadaten- und Konsistenzchecks, führt jedoch ausdrücklich
`status=reviewed_incomplete_pair` und `runtime_admitted=false`.
Die Prüfung bindet die vorhandenen gespeicherten Ergebnischecks; sie
berechnet keine Spektren erneut und ändert den negativen Paarstatus nicht.

Der frische vollständige E-Aufruf für PSD und NSD gemeinsam lieferte bei
N=1.000.000.000 tatsächlich 744 Punkte. Seine **1153,282266022 s Wall und
322,839616948 s CPU** bleiben eine einzelne **unpaarige
Referenzbeobachtung**. Es gibt keinen R3-Zeitwert und keine E/R3-Quote;
die E-Zeit ist weder ein Median noch die Hälfte eines abgeschlossenen
Optimierungsvergleichs.

Vor dem vorgesehenen R3-Kind waren **18.082.508.800 Byte** frei. Die
unveränderte Zulassung verlangte **18.147.483.648 Byte**: 16.000.000.000
Byte für den neuen Workspace plus 2.147.483.648 Byte Reserve bei schon
vorhandener Eingabe. Es fehlten **64.974.848 Byte**. Der äußere Controller
verweigerte daher den Start. Dies ist kein numerischer R3-Fehler und kein
Timeout oder Abbruch eines bereits gestarteten R3-Aufrufs.

Beide tatsächlich gestarteten Kinder endeten mit Exit 0 und wurden
eingesammelt; die eigene temporäre Ablage wurde bereinigt.
`02_direct.json`, `02_direct.log` und `02_direct.ndjson` wurden nie
angelegt. Der Parent enthält keinen Paarvergleich und keine abschließenden
paarweiten `identity_after`-/`qualification_after`-Belege, weil diese
Prüfstellen nicht erreicht wurden. E besitzt seine eigenen abgeschlossenen
Vor-/Nachprüfungen. Es wurde kein weiterer Großlauf durchgeführt.
Das bestandene kleine R3-Gate und die gesonderte fast.9-Integration
bleiben eigenständige Nachweise; sie ersetzen keine große R3-Messung.

Der unveränderte Parent-SHA256 ist
`f05ae10b8b8f7088a4c84ac72cb66c8566cf13849d7210d4b1bd48eb468355ad`.
Die getrennten Belege
`direct_log_power_r3_1b_result_review/RESULT_REVIEW.md` und
`RESULT_REVIEW.json` haben die SHA256-Werte
`22e681e64dc74d8c5d2cec72eca4abedb175a4f0173044ab758e0c3d5885c596`
beziehungsweise
`97af713c41591bf1fd96139fa7e07fb1446c173e34e19588aedffc9c390576f9`.

## Grenzen der Wiederherstellung und unveränderte Genauigkeitsgates

Die E-Matrix und die durch erhaltene Sicherungen belegten Quellen lassen
sich nachweisen. Nicht gesicherte lokale Zusatzdateien und frühere
Diagramme sind dadurch nicht automatisch wiederhergestellt. Neue
Abbildungen werden aus tatsächlich erhaltenen Rohdaten abgeleitet.
Erinnerte Zahlen oder ein früherer Dokumentationstext ersetzen keine
fehlenden Messarrays, Zeitserien, Quellenmanifeste oder Kindprozessbelege.
Wo nur eine kompakte Git-Zusammenfassung erhalten ist, bleibt diese
begrenzte Belegart ausdrücklich erkennbar.

Für den geplanten externen **Direct-log-1B-Vergleich gab es vor dem
Infrastrukturereignis noch keinen ausgeführten Lauf und keinen
Ergebnis-/Jobordner**. Ein solches Messergebnis wurde folglich nicht
verloren. Rekonstruierte Kandidatenquellen, ein kleines numerisches Gate
oder ein Buildmanifest ersetzen keinen ausgeführten Großvergleich.
Der nach der Wiederherstellung tatsächlich begonnene R3-Versuch ist oben
als vor dem Direct-Start gescheiterte, unvollständige Folge dokumentiert.
Auch eine nicht wiederhergestellte historische fs=1-Erweiterung
wird nicht aus Erinnerungszahlen nachgebildet.

Für die hier fortgeführte ursprüngliche Kandidatenqualifikation bleibt
**`rtol=3e-6`, `atol=0` für jeden PSD- und NSD-Wert** maßgeblich,
einschließlich sehr kleiner Werte und Nullen. Das frühere Packed-Real-Gate
bleibt mit 44 bestandenen und vier fehlgeschlagenen Hauptvergleichen
fehlgeschlagen. Die in bestehenden Plattformberichten beschriebene
1%-Neubewertung ist ein gesonderter Bewertungsmaßstab. Dieses Addendum
behauptet dafür keine Benutzerautorisierung zur Lockerung des Grenzwerts
und deutet keine ursprüngliche Ablehnung in eine bestandene
3-ppm-Qualifikation um.

## Neue separate Gerätebelege mit 32 GiB Benchmarkbudget

Während dieser Abschlussarbeit veröffentlichte der Hauptzweig in Commit
`3fab6be75601562444f7cd22bf0fc13c21c6204d` die separate
[physische Gerätekampagne](https://github.com/MichaDit/lpsd/blob/3fab6be75601562444f7cd22bf0fc13c21c6204d/docs/device-tests-20261012.md).
Sie enthält fast.8-Messungen bei Quellcommit
`b8c500217f7c0ad996f55b039bbf09d685800614`, unter anderem auf Windows
x86-64 mit 16 zugelassenen Workern, FFTW 3.3.5 und **32.768 MiB RAM-Budget**.
Der Rechner meldet 64.727 MiB gesamten RAM. Das ist eine tatsächlich
ausgeführte Messung mit 32 GiB Budget auf einem größeren Rechner, keine
Messung eines physisch auf 32 dezimale GB begrenzten Systems. Die hiesige
lokale E-Kampagne bleibt unter einer harten 8-GiB-Grenze.

| N / tatsächliche Punkte | Verfahren | Wall Basis, s | CPU Basis, s | Wall doppelt, s | CPU doppelt, s |
|---|---|---:|---:|---:|---:|
| 100M / 704→1408 | FFTW | 4,114062 | 6,640625 | 4,136050 | 6,859375 |
| 100M / 704→1408 | Hybrid | 19,191724 | 176,906250 | 30,617997 | 352,812500 |
| 100M / 704→1408 | LPSD/LNSD | 35,259303 | 533,593750 | 73,803126 | 1126,046875 |
| 1B / 744→1488 | FFTW | 57,379299 | 171,906250 | 55,922209 | 167,406250 |
| 1B / 744→1488 | Hybrid | 173,534242 | 915,125000 | 261,393442 | 1652,421875 |
| 1B / 744→1488 | LPSD/LNSD | 410,560081 | 5730,156250 | nicht abgeschlossen | nicht abgeschlossen |

Alle Zahlen sind einzelne abgeschlossene gemeinsame PSD+NSD-Aufrufe im
RAM-Pfad. Die Tabelle verwendet das erste Windows-Segment; nur der
1B-Hybrid mit 1488 Punkten stammt aus der dokumentierten Fortsetzung.
Deren kleinere Unterstützungsfälle werden nicht in Wiederholungsmediane
umgewandelt. Nach dem dokumentierten Hostabsturz verbleiben 131
unterschiedliche abgeschlossene Fälle von 132. Für den letzten nativen
1B-Doppelrasterfall gibt es keine fertige Zeit. Die Ursache des Absturzes
wird hier nicht dem Algorithmus zugeschrieben.

Die beiden tatsächlich gelesenen Git-Dateien
`fast8-windows-initial.json` und `fast8-windows-continuation.json`
haben SHA256
`c9c0ccbf1d906b5f0cc424a2bfe9403278165ac9799603309c9d6e6d95ad583e`
und
`eb05e6bf04af354c5c9cc9ca31776705cdbe61422b977dfe7a0256d3250faef8`.
Der [veröffentlichte Manifeststand](https://github.com/MichaDit/lpsd/blob/3fab6be75601562444f7cd22bf0fc13c21c6204d/benchmarks/results/device-tests-20261012/manifest.json)
erklärt die erhaltenen `running`-Checkpointwerte anhand der später
belegten Unterbrechungen. Eingabedigests, System, FFTW-Version, Threads und
Quellstand unterscheiden sich von E; deshalb wird kein kausaler
8→32-GiB-Speedup und kein gemeinsamer Median berechnet.

Die gleiche Publikation enthält einen abgeschlossenen ARM64-SBC-Benchmark
mit 108 erfolgreichen und 24 übersprungenen Fällen bei 384 MiB angefordertem
RAM-Budget und 2 GiB Diskbudget. Der Instrument-Checkpoint umfasst nur sechs
fast.8-Fälle mit unbekanntem späterem Abschluss. Seine Debian-ARM64-chroot
auf einem Android-Kernel belegt keinen allgemeinen Android-/Termux-Start.
Die dokumentierten unerwarteten Windows-/Instrument-Regressionstestfehler
bleiben ungelöst in dieser getrennten Gerätequelle; die folgende CI
etikettiert sie nicht nachträglich als bestandene Hardwaretests.

## Eigene Validierung der fast.9-Integration

Die optionale native Direct-log-Anbindung und die laufende Zählung der
eigenen Diskdateien sind in [PR #3](https://github.com/MichaDit/lpsd/pull/3)
implementiert. `--disk-power-reduction direct-log` gilt ausschließlich für
den ausdrücklich gewählten einfachen FFTW-Dateipfad mit voller komplexer
FFT, ursprünglichem N und dem dokumentierten festen Profil. Standard bleibt
`materialized`. Die ursprünglichen Frequenzgruppen, Float32-PSD und
Complex64-NSD-Wurzelroute sowie die unabhängige vollständige Energieprüfung
bleiben erhalten. Eingabe und vollständige FFT-Matrix erfordern weiterhin
24N logische Dateibytes; die Zulassungsreserve wurde nicht verkleinert.

Der erste Integrationscommit
`e6aa1603d01ef81b9cfb6cc03643029768712bd0` wurde lokal frisch und ohne
Fast-math gebaut. Sein vollständiger Linux-Testlauf ergab **2215 bestandene
Tests, fünf Skips und zwei erwartete Fehlschläge**: 2197 optimierte und
18 Upstream-Tests. Die JUnit-Datei enthält insgesamt 2222 Fälle; die
erwarteten Fehlschläge werden nicht zusätzlich als bestandene Fälle gezählt.
Der getrennte Direct-Smoke prüft ABI 3, kleine von null verschiedene
Leistungen, vier berichtete White-/DC-Workerfälle, den Nullenergiepfad und
die Freigabe eigener Ressourcen. Diese lokalen Belege sind an genau diesen
Commit gebunden und werden nicht als neue Milliardenmessung ausgegeben.

Seine tatsächliche erste CI endete mit **zehn erfolgreichen und zwei
fehlgeschlagenen Jobs**. Die beiden macOS-Fehler betrafen die laufende
Speicherbeobachtung: eine optionale USS-Abfrage konnte bereits gelesenes
RSS verwerfen; im Test-Extra fehlte zudem psutil. Die fehlgeschlagenen Jobs
bleiben als historische Belege erhalten.

Der Folgecommit `e0b66fe61797406e8b186d2598382d46b46bc1af` liest unter
Darwin ausschließlich RSS für den beobachteten Prozessbaum und ruft
`memory_full_info()` nicht auf. Diskfälle prüfen bei unbekanntem privatem
Zähler dieses RSS gegen dasselbe RAM-Budget. Diese konservative Prüfung
kann Dateimappings früher ablehnen. Fehlendes oder unvollständiges RSS
stoppt und reapet das Kind weiterhin nach einer Sekunde ohne verwendbare
Probe. `live_memory_rss_fallback_samples` zählt die tatsächlich verwendeten
Ersatzproben im Elternprozess; der öffentliche Sanitizer begrenzt den Wert.
Die Linux- und Windows-Budgetpolitik für gültige Proben bleibt erhalten.

Der enge Folgefix ändert fünf Dateien, die übrigen 15 Integrationsquellen
und beide vorhandenen nativen Bibliotheken sind hashgleich. Sein eigener
gezielter Linux-Testlauf mit kontrollierten Darwin-/Windows-Objekten
bestand **43 Tests ohne Skips oder Fehler**. Der unabhängige statische
Review fand keine offenen Blocker. Diese Tests sind kein Ersatz für eine
tatsächliche macOS-Ausführung; die Plattformbelege stehen getrennt im
folgenden Abschnitt.

Die tatsächliche neue CI des Folgecommits schloss mit **zwölf von zwölf
erfolgreichen Jobs** ab: [acht native Test-/Wheeljobs](https://github.com/MichaDit/lpsd/actions/runs/38240789429)
und [vier frische Bootstrapjobs](https://github.com/MichaDit/lpsd/actions/runs/38240789253).
Der Hauptjob Linux x86-64/GCC/Python 3.12 meldet 2212 bestandene optimierte
Tests, vier Skips und zwei erwartete Fehlschläge; seine Originalsuite hat
18 bestandene Tests. Der macOS-Numerikjob meldet 1819 bestandene optimierte
Tests, 397 Skips, zwei erwartete Fehlschläge und 18 bestandene Originaltests.
Die Skips betreffen optionale FFTW-/CUDA- oder Linux-spezifische Pfade. Der
separate macOS-Bootstrap installiert seinen eigenen FFTW-Pfad und hat beide
Sechs-Fälle-Smokes sowie den Direct-ABI-/Tiny-Power-/Cleanup-Test bestanden.
Die FFTW-/Disk-Teilsuiten werden nicht zusätzlich zur vollständigen Suite
addiert. Diese CI belegt ihre tatsächlich ausgeführten Plattformfälle;
sie ist keine Milliardenmessung auf jedem Zielgerät.

Die CI checkte den synthetischen PR-Merge
`d7011b46d61bd73218334ee6c358bc3dbc827640` aus. Dessen Eltern sind der neue
Hauptstand `3fab6be75601562444f7cd22bf0fc13c21c6204d` und der Folgecommit
`e0b66fe61797406e8b186d2598382d46b46bc1af`. Der separate PR-API-Basewert
lautete weiterhin `b8c500217f7c0ad996f55b039bbf09d685800614`; diese Werte
werden im Beleg nicht gleichgesetzt. Das zusätzliche Hauptzweigdelta
enthält ausschließlich README, Gerätedokumentation und veröffentlichte
Geräte-JSONs, keine Runtimeänderung.

Die [öffentliche skalare Ergänzung](../benchmarks/results/final-qualification-20261012.json)
bindet die Spalten-, R3- und ursprünglichen lokalen Integrationsnachweise.
Die [getrennte Plattformqualifikation](../benchmarks/results/fast9-platform-qualification-20261012.json)
ordnet den ersten fehlgeschlagenen CI-Stand und den Folgefix ihren eigenen
Commits zu. Rohspektren, Testlogs und private Umgebungsdaten werden hier
nicht veröffentlicht. Keine dieser Integrationsprüfungen etikettiert die
E/G-Leistungsmessungen nachträglich um.

## Portabler Benchmark und freiwillige Teilnahme

Der portable Runner und die optionale Veröffentlichung über GitHub Issues
sind bereits im Hauptzweig vorhanden; PR #3 ergänzt die oben beschriebenen
Änderungen und Nachweise. Die
[Startanleitung mit festem Quellstand](portable-benchmark.md#start-from-a-pinned-revision)
verwendet für Launcher und heruntergeladenen Quellcode denselben
qualifizierten Commit. Dieser Pin bleibt von den E/G-Messkohorten und der
externen R3-Revision getrennt. Der Pull Request wird zur Prüfung bereitgestellt;
er wird durch diese Arbeit nicht gemergt.

Messungen bleiben standardmäßig lokal. `submit RESULTS.json` bereitet den
öffentlichen Beitrag zunächst lokal vor; eine Veröffentlichung erfordert
die ausdrückliche Auswahl `--accept-public-upload`. Dafür ist kein
Repository-Schreibrecht nötig. Ein öffentliches Issue ist dem verwendeten
GitHub-Konto zugeordnet. Umfang, Vorschau und Veröffentlichung beschreibt
die bestehende [Datenschutzerklärung für Benchmarkberichte](benchmark-privacy.md).

