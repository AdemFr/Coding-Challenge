# Coding-Challenge

## Übersicht

**pipeline.py:**

Pipeline, die lokale Bilder einliest und nach der Verarbeitung in einem neuem Verzeichnis abspeichert.
Es wird "manuel" die Bounding Box um den Nagel ausgeschnitten, um eine höhere Auflösung während des Trainings zu ermöglichen.

**main.py:**

Trainingsskript für ein CNN inklusive Einlesen der Daten, Aufbau des Modells und anschließender Dokumentation.


**keras_app:**

Einfache Server App samt trainiertem Modell und Dockerfile.

Nutzung der App:

lokale Nutzung:

In:

`python model_server.py`

`curl http://localhost:5000/predict?image=<image path oder url>`

out:

`{"prediction":{"label":"good","probability":0.6206077039241791},"success":true}`

Docker image ist erhältlich auf Docker Hub unter:
`ademfr/keras_app`


