# Know_ur_feelings

Real-time emotion detection web application built with FER, Streamlit, and OpenCV. Detects and displays emotions from webcam feeds on mobile or desktop devices with a responsive and user-friendly interface.

## Features
- Detects emotions (happy, sad, angry, etc.) in real-time using the `fer` library.
- Responsive UI built with Streamlit, supporting mobile and desktop devices.
- Displays bounding boxes and emotion probabilities around detected faces.
- Includes a sidebar with usage instructions and tips.
- Deployable on Streamlit Cloud or other platforms like Render.

## Requirements
- Python 3.8+
- Virtual environment (e.g., `ai_projects`)
- Libraries: `fer`, `opencv-python`, `tensorflow`, `numpy`, `streamlit`, `torch`, `matplotlib`, `pillow`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Know_ur_feelings.git
   cd Know_ur_feelings
   ```
2. Activate your virtual environment:
   ```bash
   source ai_projects/bin/activate  # Linux/Mac
   ai_projects\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Open `http://localhost:8501` in a browser to view the emotion detection interface.
- Click "بدء الكشف عن المشاعر" to start detecting emotions from your webcam.
- On a mobile device, access the app via the deployed URL or local IP (e.g., `http://<your-ip>:8501`).
- Grant camera access when prompted.
- Use the sidebar for instructions and tips.

## Deployment
1. Push the project to GitHub.
2. Deploy on Streamlit Cloud:
   - Sign in to [share.streamlit.io](https://share.streamlit.io).
   - Create a new app and link it to the `Know_ur_feelings` repository.
   - Specify `app.py` as the main script.
   - Ensure `requirements.txt` is included.
3. Alternatively, deploy on Render:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.headless true`
4. Access the deployed URL (e.g., `https://know-ur-feelings.streamlit.app`).

## Project Structure
```
Know_ur_feelings/
├── app.py                # Streamlit application for emotion detection
├── templates/            # HTML templates (optional, for Flask integration)
│   └── index.html        # HTML template for web interface (if using Flask)
├── static/               # Static files for styling
│   └── style.css         # CSS for responsive design
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── .gitignore            # Files to ignore in version control
└── LICENSE               # MIT License
```

## Troubleshooting
- **Camera issues**: Ensure camera permissions are granted in the browser and use HTTPS for deployed apps.
- **Performance**: Reduce video resolution in `app.py` or use `FER(mtcnn=False)` for faster processing.
- **Deployment errors**: Check Streamlit Cloud or Render logs for missing dependencies.
- **Streamlit issues**: Ensure `streamlit` is updated (`pip install --upgrade streamlit`).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Screenshot
![Emotion Detection](static/images/screenshot.png)