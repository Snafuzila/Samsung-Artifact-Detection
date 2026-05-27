# 🚀 מדריך הרצה: זיהוי פגמים באמצעות דוקר (מעודכן)

לפני שמתחילים, ודא שהתמונות שאתה רוצה לבדוק נמצאות בתוך תיקיית `input` המקומית במחשב שלך.

### 🛠️ שלב הכנה (בניית ה-Image)
אם ביצעת שינויים בקוד או שזו הריצה הראשונה שלך, בנה את ה-Image ב-PowerShell/CMD:
```powershell
docker build -t samsung-detector .
```

---

## 📸 אפשרות 1: הרצה על תמונה בודדת ספציפית

### הרצה רגילה (ברירת מחדל: Threshold=0.90, IoU=0.05):
```powershell
docker run --gpus all -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" samsung-detector input/download.jpg
```

### הרצה עם ערכים מותאמים אישית (שינוי Threshold ו-IoU):
אם אתה רוצה לשנות את רמת הביטחון או סינון הכפילויות, תוכל להוסיף את הדגלים בסוף הפקודה ללא צורך בבנייה מחדש:
```powershell
docker run --gpus all -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" samsung-detector input/download.jpg --threshold 0.85 --iou 0.10
```

* 💡 **מה להחליף?** שנה את `download.jpg` לשם קובץ התמונה האמיתי שלך, ואת המספרים לערכים המבוקשים.
* 📁 **תוצאה:** קובץ ה-JSON שייווצר בתיקיית `output` ייקרא על שם התמונה, לדוגמה: `download_20260527_194512.json`.

---

## 📂 אפשרות 2: הרצה על כל תיקיית ה-input (כל התמונות במכה)

### הרצה רגילה (ברירת מחדל: Threshold=0.90, IoU=0.05):
```powershell
docker run --gpus all -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" samsung-detector input
```

### הרצה עם ערכים מותאמים אישית (שינוי Threshold ו-IoU לכל התיקייה):
הדגלים ישפיעו על כל התמונות שיעובדו בתוך הלולאה באופן אוטומטי:
```powershell
docker run --gpus all -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" samsung-detector input --threshold 0.75 --iou 0.15
```

* 💡 **איך זה עובד?** הקוד מזהה ששלחת את המילה `input` (שהיא תיקייה) ומריץ לולאה פנימית על כל הקבצים.
* 📁 **תוצאה:** עבור כל תמונה בתיקייה ייווצר קובץ JSON ייחודי משלה בתוך תיקיית `output` המקומית (למשל: `image1_timestamp.json`, `image2_timestamp.json`).

---

## ⚙️ פירוט הפרמטרים (ניתן לשילוב בכל הרצה)
* `--threshold` (ברירת מחדל `0.90`): סף הביטחון של המודל הבינארי לזיהוי חשד לפגם (ערכים בין 0 ל-1).
* `--iou` (ברירת מחדל `0.05`): סף ה-Overlap של אלגוריתם ה-NMS לצמצום קופסאות חופפות (ערכים נמוכים יותר מסננים יותר קופסאות קרובות).

📌 **הערה:** כל קבצי הפלט נשמרים ישירות בתיקיית ה-`output` במחשב שלך בזמן אמת ולא יימחקו עם סגירת הקונטיינר.
