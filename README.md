\# AI-Assisted Medical Imaging and Diagnosis System



An educational medical imaging application that uses deep learning to analyze chest X-ray images, display prediction scores, visualize model attention using Grad-CAM, and record reviewer decisions.



> \*\*Important:\*\* This project is an educational prototype, not a clinically validated medical device. It must not be used to diagnose patients or make medical decisions.



\## 1. Project Overview



The system combines a React frontend, a FastAPI backend, a deep learning model, Grad-CAM visualizations, and an audit log for reviewer decisions.



Users can upload a supported chest X-ray image, view the model's predictions, inspect a heatmap visualization, and record an approval or rejection with optional reviewer comments.



\## 2. Key Features



\* Chest X-ray image upload and preprocessing.

\* Deep learning inference using a DenseNet121-based model.

\* Prediction labels and associated scores.

\* Grad-CAM heatmap visualization.

\* Reviewer approval and rejection workflow.

\* Persistent review history using SQLite.

\* REST API built with FastAPI.

\* Interactive web interface built with React and Vite.

\* Docker-based application deployment.

\* Automated backend tests using pytest.



\## 3. Technology Stack



| Component             | Technology              |

| --------------------- | ----------------------- |

| Frontend              | React, Vite, JavaScript |

| Backend               | Python, FastAPI         |

| Machine learning      | PyTorch, DenseNet121    |

| Image processing      | Pillow, NumPy           |

| Medical image support | pydicom                 |

| Explainability        | Grad-CAM                |

| Database              | SQLite                  |

| Testing               | pytest                  |

| Deployment            | Docker, Docker Compose  |

| Web server and proxy  | Nginx                   |



\## 4. System Architecture



```text

User

&#x20;|

&#x20;v

React Frontend

&#x20;|

&#x20;v

Nginx Reverse Proxy

&#x20;|

&#x20;v

FastAPI Backend

&#x20;|

&#x20;+---- Image preprocessing

&#x20;|

&#x20;+---- Deep learning inference

&#x20;|

&#x20;+---- Grad-CAM visualization

&#x20;|

&#x20;+---- Review API

&#x20;        |

&#x20;        v

&#x20;     SQLite Database

```



Docker Compose runs the frontend and backend as separate services.



\## 5. Project Structure



```text

medical-ai-project/

├── data/

├── docs/

│   ├── model\_evaluation\_report.md

│   ├── evaluation\_v2.json

│   ├── evaluation\_v3.json

│   └── other evaluation and training records

├── frontend/

│   ├── src/

│   ├── Dockerfile

│   ├── nginx.conf

│   └── package.json

├── models/

│   └── densenet121\_chestmnist.pth

├── orthanc/

├── src/

│   ├── api.py

│   └── review\_database.py

├── tests/

│   └── test\_api.py

├── Dockerfile

├── docker-compose.yml

├── requirements.txt

└── README.md

```



The structure above highlights the main project components. Individual filenames may vary as development continues.



\## 6. Prerequisites



\* Windows, Linux, or macOS.

\* Docker Desktop or Docker Engine with the Docker Compose plugin.

\* Git, if you want to version-control the project.



Docker is the recommended way to run this application.



\## 7. Run with Docker



Clone or download the project and open a terminal in the project directory.



Build the images:



```bash

docker compose build

```



Start the application:



```bash

docker compose up -d

```



Check the running services:



```bash

docker compose ps

```



Open the application:



\* \*\*Web dashboard:\*\* http://localhost:3000

\* \*\*Backend health check:\*\* http://localhost:8000/health

\* \*\*Interactive API documentation:\*\* http://localhost:8000/docs



Stop the services:



```bash

docker compose down

```



The application uses a Docker named volume for review data so that review records can persist across container recreation.



\## 8. API Overview



The backend provides endpoints for application health, image analysis, and reviewer decisions.



| Endpoint                   | Purpose                               |

| -------------------------- | ------------------------------------- |

| `GET /health`              | Check backend health and model status |

| `POST /reviews`            | Record an approval or rejection       |

| `GET /reviews`             | Retrieve review history               |

| `GET /reviews/{review\_id}` | Retrieve a specific review            |



For the complete request and response schemas, start the application and visit http://localhost:8000/docs.



\## 9. Automated Testing



The backend uses pytest for automated API tests.



Run the tests locally:



```bash

python -m pytest -v

```



Run them inside the Docker backend container:



```bash

docker compose exec backend python -m pytest -v

```



\*\*Recorded test result:\*\* 7 tests passed inside the Docker backend container during development.



Re-run the tests after making changes to verify the current code.



\## 10. Model Evaluation



The `docs/` directory contains model evaluation reports, training histories, threshold-tuning records, and model comparison results.



These records should be consulted when interpreting the model's performance.



The model is experimental. Its predictions and confidence scores should not be interpreted as clinically reliable probabilities without appropriate validation and calibration.



\## 11. Limitations and Responsible Use



\* The current model is intended for educational experimentation.

\* The model was developed for the ChestMNIST dataset and has not been validated for general clinical use.

\* Performance on one dataset does not establish performance across hospitals, populations, scanners, or imaging protocols.

\* Grad-CAM highlights image regions associated with model activations; it does not prove that the model's reasoning is medically correct.

\* The application should not be used to diagnose, treat, or triage patients.

\* Real clinical deployment would require appropriate clinical validation, privacy and security controls, regulatory review, and expert oversight.

\* The current implementation should not be described as a validated CT or MRI analysis system.



\## 12. Future Improvements



Potential future work includes:



\* Improve model evaluation and document dataset limitations.

\* Add stronger input validation and error handling.

\* Expand automated tests.

\* Improve audit logging and reviewer authentication.

\* Add secure access controls and privacy safeguards.

\* Integrate a DICOM server such as Orthanc.

\* Evaluate suitable segmentation models using validated datasets.

\* Improve deployment monitoring and resource efficiency.



These are future enhancements, not claims that all features are currently implemented.



\## 13. License and Dataset Attribution



Add a license before distributing this project publicly.



If model weights or datasets are redistributed, follow their applicable licensing and attribution requirements. Do not include patient-identifiable medical data or credentials in a public repository.



\## 14. Disclaimer



This software is provided for educational and research purposes only. It is not intended for clinical diagnosis or treatment and has not been established as safe or effective for medical decision-making.
