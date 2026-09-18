# King of Expressions — Facial Expression Controller 👑

A real-time AI facial expression detector and meme generator that runs entirely inside your web browser using **MediaPipe Face Mesh** and **Hands**.

It tracks your face landmarks in real time and automatically mirrors your expression with the matching meme.

---

## 🎭 The 9 Supported Expressions (Matching Asset Images)

| Expression | Asset Image | Trigger Facial Feature / Gesture |
|---|---|---|
| 😏 **CHILIPI** | `chilipi.png` | Two eyebrows raised + squinted eyes + closed smiling mouth + nose squeezed |
| 🤨 **DOUBT** | `doubt.png` | Squinting eyes with hand covering or near mouth |
| 😄 **HAPPY** | `happy.png` | Wide smile with corners turned upwards |
| 😐 **NEUTRAL** | `neutral.png` | Relaxed, resting facial expression |
| 🤪 **PICHA HAPPY** | `picha happy.png` | Wide laughing open mouth + high smile |
| 😎 **PROUD** | `proud.png` | Confident proud expression / head held high |
| 😲 **SHOCK** | `shock.png` | Mouth wide open + eyes wide + raised eyebrows |
| 😳 **SIGGU** | `siggu.png` | Head turned sideways / shy smile looking away |
| 🤔 **THINKING** | `thinking.png` | Hand resting under or against your chin |

---

## 🌐 Web Applications Included

1. **Main Controller (`index.html`)**:
   - Live AI classifier powered by your trained profile with instant meme reactions.
2. **AI Custom Trainer (`trainer.html`)**:
   - Single-click 200 samples capture per expression with auto-stop.
   - Per-expression reset option.
   - **Train each expression with your own unique face!**
   - Click or hold **Spacebar / "Hold to Capture"** to record frames of your face for any expression.
   - Powered by a real-time **k-Nearest Neighbors (k-NN)** classifier running directly in your browser.
   - Save your trained profile to `localStorage`, or export/import as JSON files.
   - Live confidence score and side-by-side meme matching.

---

## 🚀 How to Run Locally

Because browser camera access (`getUserMedia`) requires a secure origin (`http://localhost` or `https://`):

```bash
cd wojak-expression-controller
python server.py
```
- Open **`http://localhost:8000`** for the Main Controller.
- Open **`http://localhost:8000/trainer.html`** for the Custom AI Trainer.
- Hidden photos captured during sessions are automatically saved directly into the **`people/`** folder.

### Free Web Hosting (Netlify / Vercel / GitHub Pages)
You can deploy both pages publicly in seconds:
- **Netlify**: Drag and drop the `wojak-expression-controller` folder onto [Netlify Drop](https://app.netlify.com/drop).
- **Vercel**: Import the folder as a static site.
- **GitHub Pages**: Push this repository to GitHub and enable Pages in repository settings.
