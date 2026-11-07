import { useEffect, useState } from 'react'
import './App.css'

const API_URL = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '')
const MAX_FILE_SIZE = 10 * 1024 * 1024

const IMAGE_TYPES = [
  'image/png',
  'image/jpeg',
  'image/webp',
]

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [predictions, setPredictions] = useState([])
  const [gradcamImage, setGradcamImage] = useState('')
  const [gradcamLabel, setGradcamLabel] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [sourceFormat, setSourceFormat] = useState('')

  // Review workflow
  const [reviewId, setReviewId] = useState('')
  const [reviewFilename, setReviewFilename] = useState('')
  const [reviewer, setReviewer] = useState('')
  const [comments, setComments] = useState('')
  const [savingReview, setSavingReview] = useState(false)
  const [reviewSaved, setReviewSaved] = useState(false)
  const [reviewMessage, setReviewMessage] = useState('')
  const [reviewError, setReviewError] = useState('')

  // Review history
  const [reviews, setReviews] = useState([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview)
      }
    }
  }, [preview])

  useEffect(() => {
    loadReviewHistory()
  }, [])

  async function loadReviewHistory() {
    setHistoryLoading(true)
    setHistoryError('')

    try {
      const response = await fetch(`${API_URL}/reviews?limit=50`)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail || 'Unable to load review history.'
        )
      }

      setReviews(
        Array.isArray(data.reviews) ? data.reviews : []
      )
    } catch (err) {
      setHistoryError(
        `${err.message} Check that the backend is running.`
      )
    } finally {
      setHistoryLoading(false)
    }
  }

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0]

    setPredictions([])
    setGradcamImage('')
    setGradcamLabel('')
    setError('')
    setNotice('')
    setSourceFormat('')
    setFile(null)
    setPreview('')

    setReviewId('')
    setReviewFilename('')
    setComments('')
    setReviewSaved(false)
    setReviewMessage('')
    setReviewError('')

    if (!selectedFile) return

    const fileName = selectedFile.name.toLowerCase()

    const isDicom =
      fileName.endsWith('.dcm') ||
      fileName.endsWith('.dicom')

    const isImage = IMAGE_TYPES.includes(selectedFile.type)

    if (!isImage && !isDicom) {
      setError(
        'Unsupported file. Select PNG, JPG, WebP, or DICOM.'
      )
      event.target.value = ''
      return
    }

    if (selectedFile.size === 0) {
      setError('The selected file is empty.')
      event.target.value = ''
      return
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setError('The file must be 10 MB or smaller.')
      event.target.value = ''
      return
    }

    setFile(selectedFile)

    if (isImage) {
      setPreview(URL.createObjectURL(selectedFile))
      setNotice('Image selected and ready for experimental analysis.')
    } else {
      setNotice(
        'DICOM file selected. The Python API will process this file.'
      )
    }
  }

  async function analyzeImage() {
    if (!file) {
      setError('Please select an image or DICOM file first.')
      return
    }

    setLoading(true)
    setError('')
    setPredictions([])
    setGradcamImage('')
    setGradcamLabel('')
    setSourceFormat('')
    setReviewId('')
    setReviewFilename('')
    setReviewSaved(false)
    setReviewMessage('')
    setReviewError('')
    setNotice('')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      const responseText = await response.text()

      let data

      try {
        data = responseText ? JSON.parse(responseText) : {}
      } catch {
        throw new Error(
          `Invalid API response (HTTP ${response.status}).`
        )
      }

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'Unable to analyze the selected file.'
        )
      }

      const results = (
        Array.isArray(data.predictions) ? data.predictions : []
      )
        .map((item) => ({
          label: item.label,
          probability: Number(item.probability),
        }))
        .filter(
          (item) =>
            item.label &&
            Number.isFinite(item.probability)
        )
        .sort((a, b) => b.probability - a.probability)

      if (!results.length) {
        throw new Error(
          'The API did not return valid prediction scores.'
        )
      }

      if (!data.review_id) {
        throw new Error(
          'The API did not return a review ID.'
        )
      }

      setPredictions(results)
      setReviewId(data.review_id)
      setReviewFilename(data.filename || file.name)
      setGradcamImage(data.gradcam_image || '')
      setGradcamLabel(data.gradcam_label || '')
      setSourceFormat(data.source_format || '')
      setNotice(
        data.notice ||
          'Educational prototype only. Results are not a medical diagnosis.'
      )
    } catch (err) {
      setError(
        `${err.message} Check that the Python API and frontend proxy are running.`
      )
    } finally {
      setLoading(false)
    }
  }

  async function submitReview(decision) {
    if (!reviewId || !reviewFilename) {
      setReviewError(
        'Run an analysis before submitting a review.'
      )
      return
    }

    if (!reviewer.trim()) {
      setReviewError('Enter the reviewer name first.')
      return
    }

    if (reviewSaved) {
      setReviewError(
        'A review has already been saved for this analysis.'
      )
      return
    }

    setSavingReview(true)
    setReviewError('')
    setReviewMessage('')

    try {
      const response = await fetch(`${API_URL}/reviews`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          review_id: reviewId,
          filename: reviewFilename,
          decision,
          reviewer: reviewer.trim(),
          comments: comments.trim(),
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail || 'Unable to save the review.'
        )
      }

      setReviewSaved(true)
      setReviewMessage(
        `Review ${decision} successfully saved.`
      )

      await loadReviewHistory()
    } catch (err) {
      setReviewError(err.message)
    } finally {
      setSavingReview(false)
    }
  }

  const isDicomFile = file
    ? /\.(dcm|dicom)$/i.test(file.name)
    : false

  function formatDate(value) {
    if (!value) return 'Date unavailable'

    const date = new Date(value)

    if (Number.isNaN(date.getTime())) {
      return value
    }

    return date.toLocaleString()
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#">
          <span className="brand-icon">✚</span>

          <span>
            <strong>MedVision AI</strong>
            <small>Medical Imaging Research Platform</small>
          </span>
        </a>

        <div className="system-status">
          <span className="status-dot" />
          Educational Prototype
        </div>
      </header>

      <section className="hero-section">
        <div className="hero-copy">
          <span className="eyebrow">
            AI-ASSISTED MEDICAL IMAGING
          </span>

          <h1>
            Medical imaging,
            <br />
            <span>made easier to explore.</span>
          </h1>

          <p>
            Upload an image or DICOM file to explore predictions
            from an experimental deep-learning model. Results
            are for educational purposes only.
          </p>
        </div>

        <div className="hero-symbol" aria-hidden="true">
          <div className="scan-frame">
            <span className="scan-cross">✚</span>
            <span className="scan-ring ring-one" />
            <span className="scan-ring ring-two" />
            <span className="scan-ring ring-three" />
            <span className="scan-center" />
          </div>

          <span className="symbol-caption">
            AI · IMAGING · RESEARCH
          </span>
        </div>
      </section>

      <section className="notice-banner">
        <span className="notice-icon">!</span>

        <div>
          <strong>Research demonstration only</strong>

          <p>
            This model is not clinically validated. Do not use its
            results to diagnose, treat, or make medical decisions.
          </p>
        </div>
      </section>

      <section className="workspace">
        <div className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <span className="step-label">STEP 01</span>
              <h2>Upload scan</h2>
              <p>Select an image or DICOM file for analysis.</p>
            </div>

            <span className="panel-icon">↑</span>
          </div>

          <label className="upload-zone">
            {preview ? (
              <img
                className="image-preview"
                src={preview}
                alt="Selected scan preview"
              />
            ) : file ? (
              <span className="upload-placeholder">
                <span className="upload-icon">✓</span>
                <strong>
                  {isDicomFile
                    ? 'DICOM file selected'
                    : 'Image selected'}
                </strong>
                <span>{file.name}</span>
                <small>Ready for experimental analysis.</small>
              </span>
            ) : (
              <span className="upload-placeholder">
                <span className="upload-icon">↑</span>
                <strong>Choose a scan</strong>
                <span>Click here to browse your computer</span>
                <small>
                  PNG, JPG, WebP, DICOM · Maximum 10 MB
                </small>
              </span>
            )}

            <input
              type="file"
              accept="image/png,image/jpeg,image/webp,.dcm,.dicom,application/dicom"
              onChange={handleFileChange}
              hidden
            />
          </label>

          {file && (
            <div className="file-information">
              <span className="file-icon">▧</span>

              <span className="file-details">
                <strong>{file.name}</strong>
                <small>
                  {isDicomFile ? 'DICOM file' : 'Image file'}
                  {' · '}
                  {(file.size / 1024).toFixed(1)} KB
                </small>
              </span>

              <span className="file-check">✓</span>
            </div>
          )}

          <button
            className="primary-button"
            type="button"
            onClick={analyzeImage}
            disabled={!file || loading}
          >
            {loading ? 'Analyzing file...' : 'Run AI analysis →'}
          </button>

          {error && (
            <p className="error-message" role="alert">
              {error}
            </p>
          )}

          {notice && !predictions.length && !loading && !error && (
            <p className="api-notice">{notice}</p>
          )}

          <div className="upload-footnote">
            <span>🔒</span>
            <span>
              Use sample files without personal or identifiable
              patient information.
            </span>
          </div>
        </div>

        <div className="panel results-panel">
          <div className="panel-heading">
            <div>
              <span className="step-label">STEP 02</span>
              <h2>Analysis results</h2>
              <p>Experimental outputs returned by the model.</p>
            </div>

            <span className="panel-icon">⌁</span>
          </div>

          {!predictions.length && !loading && (
            <div className="empty-results">
              <div className="empty-illustration">
                <span>▤</span>
                <span className="empty-search">⌕</span>
              </div>

              <h3>Waiting for a scan</h3>
              <p>
                Select a file and run the analysis to display
                the model's output here.
              </p>
            </div>
          )}

          {loading && (
            <div className="empty-results">
              <div className="loading-spinner" />
              <h3>Analyzing file</h3>
              <p>Please wait while the model processes your file.</p>
            </div>
          )}

          {predictions.length > 0 && (
            <div className="prediction-results">
              <div className="results-summary">
                <span className="results-check">✓</span>

                <div>
                  <strong>Analysis complete</strong>
                  <small>
                    {predictions.length} model outputs received
                  </small>
                </div>
              </div>

              {sourceFormat && (
                <p>
                  <strong>Input format:</strong> {sourceFormat}
                </p>
              )}

              <h3 className="results-subheading">
                Model output scores
              </h3>

              {predictions.map((item) => {
                const percentage = Math.max(
                  0,
                  Math.min(100, item.probability * 100)
                )

                return (
                  <div
                    className="prediction-row"
                    key={item.label}
                  >
                    <div className="prediction-label">
                      <span>{item.label}</span>
                      <strong>{percentage.toFixed(1)}%</strong>
                    </div>

                    <div
                      className="probability-track"
                      role="progressbar"
                      aria-label={`${item.label} model output`}
                      aria-valuenow={percentage}
                      aria-valuemin="0"
                      aria-valuemax="100"
                    >
                      <div
                        className="probability-fill"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                )
              })}

              <div className="result-disclaimer">
                <strong>Interpretation caution</strong>
                <p>
                  These values are experimental model outputs,
                  not confirmed disease probabilities or a
                  medical diagnosis.
                </p>
              </div>
            </div>
          )}

          {gradcamImage && predictions.length > 0 && (
            <div className="gradcam-section">
              <h3>Grad-CAM Visual Explanation</h3>

              <p>
                Experimental heatmap for:{' '}
                <strong>
                  {gradcamLabel || 'Model output'}
                </strong>
              </p>

              <img
                src={gradcamImage}
                alt={`Experimental Grad-CAM heatmap for ${gradcamLabel || 'model output'}`}
                style={{
                  display: 'block',
                  width: '100%',
                  maxWidth: '420px',
                  height: 'auto',
                  margin: '16px auto',
                  borderRadius: '12px',
                }}
              />

              <p>
                Highlighted regions represent areas influencing
                the model output. This visualization is experimental
                and is not a medical diagnosis.
              </p>
            </div>
          )}

          {notice && predictions.length > 0 && (
            <p className="api-notice">{notice}</p>
          )}

          {/* STEP 03: Reviewer decision */}
          {predictions.length > 0 && (
            <section className="review-form">
              <div className="panel-heading">
                <div>
                  <span className="step-label">STEP 03</span>
                  <h2>Reviewer decision</h2>
                  <p>
                    Record an educational demonstration review.
                  </p>
                </div>
              </div>

              <div className="review-id-box">
                <strong>Review ID</strong>
                <code>{reviewId}</code>
              </div>

              <label className="form-label" htmlFor="reviewer">
                Reviewer name *
              </label>

              <input
                id="reviewer"
                className="review-input"
                type="text"
                maxLength={100}
                value={reviewer}
                onChange={(event) => setReviewer(event.target.value)}
                placeholder="Enter reviewer name"
                disabled={reviewSaved || savingReview}
              />

              <label className="form-label" htmlFor="comments">
                Comments
              </label>

              <textarea
                id="comments"
                className="review-input review-textarea"
                maxLength={2000}
                value={comments}
                onChange={(event) => setComments(event.target.value)}
                placeholder="Add comments about this demonstration (optional)"
                rows={4}
                disabled={reviewSaved || savingReview}
              />

              <p className="review-help">
                This is an audit-log demonstration, not a clinical
                approval workflow.
              </p>

              <div className="review-actions">
                <button
                  className="review-button approve-button"
                  type="button"
                  onClick={() => submitReview('approved')}
                  disabled={savingReview || reviewSaved}
                >
                  {savingReview ? 'Saving...' : '✓ Approve'}
                </button>

                <button
                  className="review-button reject-button"
                  type="button"
                  onClick={() => submitReview('rejected')}
                  disabled={savingReview || reviewSaved}
                >
                  {savingReview ? 'Saving...' : '✕ Reject'}
                </button>
              </div>

              {reviewMessage && (
                <p className="success-message" role="status">
                  {reviewMessage}
                </p>
              )}

              {reviewError && (
                <p className="error-message" role="alert">
                  {reviewError}
                </p>
              )}
            </section>
          )}
        </div>
      </section>

      {/* STEP 04: Audit history */}
      <section className="panel history-panel">
        <div className="panel-heading">
          <div>
            <span className="step-label">STEP 04</span>
            <h2>Review history</h2>
            <p>
              Previously saved reviewer decisions from the SQLite
              audit database.
            </p>
          </div>

          <button
            className="refresh-button"
            type="button"
            onClick={loadReviewHistory}
            disabled={historyLoading}
          >
            {historyLoading ? 'Refreshing...' : '↻ Refresh'}
          </button>
        </div>

        {historyError && (
          <p className="error-message" role="alert">
            {historyError}
          </p>
        )}

        {historyLoading && reviews.length === 0 && (
          <p className="history-empty">Loading review history...</p>
        )}

        {!historyLoading && !historyError && reviews.length === 0 && (
          <p className="history-empty">
            No reviews saved yet. Analyze a file and submit a
            reviewer decision to see it here.
          </p>
        )}

        {reviews.length > 0 && (
          <div className="history-table-wrapper">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Decision</th>
                  <th>Reviewer</th>
                  <th>Comments</th>
                  <th>Date</th>
                  <th>Review ID</th>
                </tr>
              </thead>

              <tbody>
                {reviews.map((review) => (
                  <tr key={review.id || review.review_id}>
                    <td>{review.filename}</td>

                    <td>
                      <span
                        className={`decision-badge ${
                          review.decision === 'approved'
                            ? 'decision-approved'
                            : 'decision-rejected'
                        }`}
                      >
                        {review.decision}
                      </span>
                    </td>

                    <td>{review.reviewer}</td>
                    <td>{review.comments || '—'}</td>
                    <td>{formatDate(review.created_at)}</td>

                    <td>
                      <code>{review.review_id}</code>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <p className="history-footnote">
          Showing up to 50 recent reviews. Records are stored in
          <code> data/review_audit.db</code>.
        </p>
      </section>

      <footer className="footer">
        <div>
          <strong>MedVision AI</strong>
          <span>
            AI-Assisted Medical Imaging and Diagnosis System
          </span>
        </div>

        <span>Educational use only · Not for clinical use</span>
      </footer>
    </main>
  )
}

export default App