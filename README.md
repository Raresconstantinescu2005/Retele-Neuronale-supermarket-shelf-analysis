# Retele-Neuronale-supermarket-shelf-analysis

🚀 Progres Proiect: Sistem de Analiză Rafturi (End-to-End)
📅 Update: Implementare Pipeline Antrenare & Interfață Web
În această etapă, am finalizat arhitectura completă a proiectului, de la preprocesarea datelor brute până la vizualizarea rezultatelor într-o aplicație web.

✅ 1. Procesarea Datelor & Antrenare (Backend AI)
Structurare Dataset: Organizarea datelor în format standard YOLOv8 (train, validation, test) cu subfoldere dedicate pentru imagini și etichete.

Preprocesare Automată: Implementarea scripturilor Python pentru redimensionarea automată a imaginilor la 640x640 px și organizarea lor în folderul processed, păstrând originalele intacte în raw.

Configurare Antrenare:

Crearea fișierului data.yml cu căile absolute pentru a evita erorile de sistem.

Dezvoltarea scriptului robust train.py care verifică existența modelelor anterioare (best.pt) pentru a preveni suprascrierea accidentală și optimizează resursele (workers=0 pentru Windows).

Model: Antrenarea unei rețele YOLOv8 Nano pe setul de date SKU-110K (adaptat) pentru detecția produselor la raft.

💻 2. Interfață Grafică Web (Frontend)
Am dezvoltat o aplicație web modernă pentru a permite utilizatorilor să testeze modelul fără a scrie cod.

Tehnologie: Aplicație bazată pe Flask (Python).

Funcționalități:

Upload de imagini (Drag & Drop).

Procesare în timp real folosind modelul antrenat (best.pt).

Afișarea rezultatului cu Bounding Boxes desenate peste produse.

Generarea unui raport sumar pe zonele raftului (Sus/Mijloc/Jos).

Design (UI/UX): Interfață stilizată cu CSS personalizat, având un aspect curat și profesional (Carduri pentru rezultate, butoane stilizate, layout responsiv).
