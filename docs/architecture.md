\# System Architecture



\## 1. Overview



The AI-Assisted Medical Imaging and Diagnosis System is an educational web application that combines a React frontend, a FastAPI backend, a deep learning model, Grad-CAM visualization, and a database for recording reviewer decisions.



The application is containerized with Docker Compose.



\## 2. Architecture Diagram



```text

+---------------------------+

|          User             |

| Uploads X-ray and reviews |

+-------------+-------------+

&#x20;             |

&#x20;             v

+---------------------------+

| React Frontend            |

| Image upload and results  |

| Prediction and heatmap UI |

| Reviewer decision form    |

+-------------+-------------+

&#x20;             |

&#x20;             v

+---------------------------+

| Nginx                     |

| Serves frontend           |

| Proxies /api/ requests    |

+-------------+-------------+

&#x20;             |

&#x20;             v

+---------------------------+

| FastAPI Backend           |

| Input validation          |

| Image preprocessing       |

| Model inference           |

| Grad-CAM generation       |

| Review API                |

+-------+-------------+-----+

&#x20;       |             |

&#x20;       v             v

+---------------+ +----------------+

| DenseNet121   | | SQLite Database|

| Model weights | | Review audit   |

+---------------+ +----------------+

```



\## 3. Frontend



The frontend is implemented using React and Vite.



Responsibilities include:



\* Uploading supported chest X-ray images.

\* Sending requests to the backend.

\* Displaying model predictions and scores.

\* Showing Grad-CAM visualizations.

\* Allowing reviewers to record approval or rejection decisions.

\* Displaying review history.



The production frontend is served by Nginx inside a Docker container.



\## 4. Backend



The backend uses Python and FastAPI.



Responsibilities include:



\* Accepting and validating API requests.

\* Preprocessing uploaded images.

\* Running inference with the configured model.

\* Generating Grad-CAM visualizations.

\* Exposing endpoints for reviewer decisions.

\* Returning review history.

\* Providing a health-check endpoint.



The backend listens on port 8000 inside its container.



\## 5. Machine Learning Pipeline



The image analysis pipeline consists of:



1\. Receive a supported image.

2\. Validate the uploaded file.

3\. Convert and preprocess the image.

4\. Resize and normalize it as required by the model.

5\. Run inference using the configured DenseNet121-based model.

6\. Return prediction labels and scores.

7\. Generate a Grad-CAM visualization when supported by the request.



The configured model was developed for ChestMNIST. It is experimental and is not clinically validated.



\## 6. Review Audit Database



The review functionality stores reviewer decisions in SQLite.



Recorded information includes:



\* Review identifier.

\* Image filename.

\* Decision.

\* Reviewer name.

\* Optional comments.

\* Creation timestamp.



The Docker Compose configuration uses a named volume for the database directory so that review data can persist across container recreation.



The audit log records review actions but does not independently establish the accuracy of a model prediction.



\## 7. Docker Deployment



The system runs as two primary services:



\### Backend



\* Built from the root Dockerfile.

\* Runs FastAPI through Uvicorn.

\* Exposes port 8000.

\* Uses a health check to report service readiness.

\* Mounts a named volume for review data.



\### Frontend



\* Built from `frontend/Dockerfile`.

\* Uses a Node.js build stage to generate production assets.

\* Serves the built application through Nginx.

\* Exposes port 3000 on the host.

\* Proxies `/api/` requests to the backend service.



Docker Compose coordinates the services.



\## 8. Testing



Automated backend tests are stored in the `tests/` directory and executed with pytest.



During development, seven tests passed inside the Docker backend container.



Tests should be rerun after application changes.



\## 9. Security and Limitations



This is an educational prototype, not a clinical system.



Before any real-world deployment, additional work would be needed for:



\* Authentication and authorization.

\* Secure handling of sensitive medical data.

\* Production logging and monitoring.

\* Model validation across representative populations.

\* Clinical evaluation and regulatory review.

\* Reliable backup and recovery procedures.



Grad-CAM visualizations are explanatory aids and do not prove that predictions are correct.



\## 10. Intended Use



The application is intended for educational demonstrations and software development practice only. It must not be used to diagnose or treat patients.
