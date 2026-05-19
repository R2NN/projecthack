# Внешние артефакты

Эта папка намеренно не версионируется целиком. Сюда кладутся данные и веса, которые нельзя хранить в публичном GitHub-репозитории:

- `db_hack.csv` — точный каталог товаров;
- `models/rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth` — обученный чекпойнт детектора;
- дополнительные локальные модели OCR, если они используются в конкретном стенде.

Минимальная структура для полного инференса:

```text
artifacts/
  db_hack.csv
  models/
    rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth
```

Проверить, что артефакты подключены:

```powershell
python tools/check_artifacts.py
```
