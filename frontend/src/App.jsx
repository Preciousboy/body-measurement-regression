import { useState } from "react";

const API_URL = import.meta.env.API_URL;

function App() {
  const [frontImage, setFrontImage] = useState(null);
  const [sideImage, setSideImage] = useState(null);
  const [frontPreview, setFrontPreview] = useState(null);
  const [sidePreview, setSidePreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleImageChange = (event, type) => {
    const file = event.target.files[0];

    if (!file) return;

    if (type === "front") {
      setFrontImage(file);
      setFrontPreview(URL.createObjectURL(file));
    } else {
      setSideImage(file);
      setSidePreview(URL.createObjectURL(file));
    }

    setResults(null);
    setError("");
  };

  const predictMeasurements = async () => {
    if (!frontImage || !sideImage) {
      setError("Please upload both a front and side photo.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);

    const formData = new FormData();

    formData.append("front_image", frontImage);
    formData.append("side_image", sideImage);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data = await response.json();

      setResults(data);
    } catch (err) {
      setError(
        "Unable to get prediction. Please check that the API is running.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <h1>BodyM Regression system</h1>

        <p style={styles.description}>
          Upload a front and side photo to estimate body measurements.
        </p>
        <p style={styles.alert}>
          Make sure you upload an image showing your full height just like the
          images beside front photo and side photo
        </p>

        <div style={styles.uploadContainer}>
          {/* Front Photo */}
          <div style={styles.uploadBox}>
            <div style={styles.photo}>
              <h3>Front Photo</h3>
              <img src="front.png" alt="front image" style={styles.image} />
            </div>

            {frontPreview ? (
              <img
                src={frontPreview}
                alt="Front preview"
                style={styles.preview}
              />
            ) : (
              <div style={styles.placeholder}>
                <div>
                  <p>No photo selected</p>
                  <p style={styles.alert}>Image format jpeg, jpg, png, webp</p>
                </div>
              </div>
            )}

            <label style={styles.uploadButton}>
              Choose Front Photo
              <input
                type="file"
                accept="image/jpeg,image/jpg,image/png,image/webp"
                onChange={(e) => handleImageChange(e, "front")}
                style={{ display: "none" }}
              />
            </label>
          </div>

          {/* Side Photo */}
          <div style={styles.uploadBox}>
            <div style={styles.photo}>
              <h3>Side Photo</h3>
              <img src="side.png" alt="front image" style={styles.image} />
            </div>

            {sidePreview ? (
              <img
                src={sidePreview}
                alt="Side preview"
                style={styles.preview}
              />
            ) : (
              <div style={styles.placeholder}>
                <div>
                  <p>No photo selected</p>
                  <p style={styles.alert}>Image format jpeg, jpg, png, webp</p>
                </div>
              </div>
            )}

            <label style={styles.uploadButton}>
              Choose Side Photo
              <input
                type="file"
                accept="image/jpeg,image/jpg,image/png,image/webp"
                onChange={(e) => handleImageChange(e, "side")}
                style={{ display: "none" }}
              />
            </label>
          </div>
        </div>

        {/* Predict Button */}
        <button
          onClick={predictMeasurements}
          disabled={loading}
          style={styles.predictButton}
        >
          {loading ? "Fetching measurement..." : "Get Measurements"}
        </button>

        {/* Error */}
        {error && <p style={styles.error}>{error}</p>}

        {/* Results */}
        {results && (
          <div style={styles.results}>
            <h2>Predicted Measurements</h2>

            <div style={styles.measurements}>
              {Object.entries(results).map(([name, value]) => (
                <div style={styles.measurement} key={name}>
                  <span>{name.replaceAll("-", " ")}</span>

                  <strong>{Number(value).toFixed(2)} cm</strong>
                </div>
              ))}
            </div>
          </div>
        )}

        <p style={styles.footer}>
          BodyM • Machine Learning Body Measurement System
        </p>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#f5f5f5",
    padding: "40px 20px",
    fontFamily: "Arial, sans-serif",
  },

  container: {
    maxWidth: "900px",
    margin: "auto",
    background: "white",
    padding: "40px",
    borderRadius: "16px",
    boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
    textAlign: "center",
  },

  // title: {
  //   fontSize: "42px",
  //   marginBottom: "25px",
  // },

  alert: {
    fontSize: "15px",
    color: "red",
    marginTop: "0",
  },

  description: {
    color: "#666",
    marginBottom: "10px",
  },

  uploadContainer: {
    display: "flex",
    gap: "25px",
    justifyContent: "center",
    flexWrap: "wrap",
  },

  uploadBox: {
    width: "320px",
    border: "2px dashed #ccc",
    borderRadius: "12px",
    padding: "20px",
  },

  photo: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "30px",
  },
  image: {
    width: "40px",
    height: "60px",
  },

  placeholder: {
    height: "220px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#f5f5f5",
    color: "#888",
    borderRadius: "8px",
    marginBottom: "15px",
  },

  preview: {
    width: "100%",
    height: "220px",
    objectFit: "contain",
    background: "#f5f5f5",
    borderRadius: "8px",
    marginBottom: "15px",
  },

  uploadButton: {
    display: "inline-block",
    padding: "10px 16px",
    background: "#222",
    color: "white",
    borderRadius: "8px",
    cursor: "pointer",
  },

  predictButton: {
    marginTop: "35px",
    padding: "14px 30px",
    fontSize: "16px",
    border: "none",
    borderRadius: "8px",
    background: "#111",
    color: "white",
    cursor: "pointer",
  },

  error: {
    color: "red",
    marginTop: "20px",
  },

  results: {
    marginTop: "40px",
    textAlign: "left",
  },

  measurements: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
    gap: "12px",
  },

  measurement: {
    display: "flex",
    justifyContent: "space-between",
    padding: "15px",
    background: "#f5f5f5",
    borderRadius: "8px",
    textTransform: "capitalize",
  },

  footer: {
    marginTop: "40px",
    color: "#888",
    fontSize: "14px",
  },
};

export default App;
