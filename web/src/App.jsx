import { useState, useRef, useEffect } from "react";
import "./App.css";

const DATES = {
    "Wednesday, March 4": "2026_03_04",
    "Thursday, March 5": "2026_03_05",
    "Friday, March 6": "2026_03_06",
    "Saturday, March 7": "2026_03_07",
    "Sunday, March 8": "2026_03_08",
    "Monday, March 9": "2026_03_09",
    "Tuesday, March 10": "2026_03_10",
    "Wednesday, March 11": "2026_03_11",
    "Thursday, March 12": "2026_03_12",
    "Friday, March 13": "2026_03_13",
    "Saturday, March 14": "2026_03_14",
    "Sunday, March 15": "2026_03_15",
};

const SONIFICATION_TYPES = { "Active Buses": "buscount", "Average Speed": "avgspeed" };

const BIN = "30s";

export default function App() {
    const [date, setDate] = useState(DATES["Wednesday, March 4"]);
    const [sonification, setSonification] = useState(SONIFICATION_TYPES["Active Buses"]);
    const [notFound, setNotFound] = useState(false);
    const videoRef = useRef(null);

    const videoSrc = `/renders/${date}_${sonification}_${BIN}.mp4`;

    // Reset not-found state and reload video when selection changes
    useEffect(() => {
        setNotFound(false);
        if (videoRef.current) {
            videoRef.current.load();
        }
    }, [videoSrc]);

    return (
        <div className="container-fluid vh-100 d-flex flex-column py-3 bg-dark text-light overflow-hidden">
            <h4 className="mb-3 text-center fw-semibold">CapMetro Sonification Gallery</h4>
            <div className="row flex-grow-1 g-3 align-items-stretch">
                {/* Video panel */}
                <div className="col-7 d-flex align-items-center justify-content-center rounded">
                    <div className="video-frame">
                        {notFound ? (
                            <div className="text-center text-secondary">
                                <p className="fs-5 mb-1">Not yet rendered</p>
                                <p className="small text-muted">
                                    Run{" "}
                                    <code>
                                        python matplot-map.py {date} --prerender --sonification{" "}
                                        {sonification}
                                    </code>
                                </p>
                            </div>
                        ) : (
                            <video
                                ref={videoRef}
                                className="w-100"
                                style={{ objectFit: "contain" }}
                                controls
                                autoPlay
                                loop
                                onError={() => setNotFound(true)}
                            >
                                <source src={videoSrc} type="video/mp4" />
                            </video>
                        )}
                    </div>
                </div>

                {/* Controls panel */}
                <div className="col-5">
                    <div className="card-body">
                        {/* Wrap selectors in a d-flex container */}
                        <div className="d-flex gap-3 align-items-start">
                            <div className="mb-3 flex-grow-1">
                                <label htmlFor="date-select" className="form-label fw-medium">
                                    Date
                                </label>
                                <select
                                    id="date-select"
                                    className="form-select"
                                    value={date}
                                    onChange={(e) => setDate(e.target.value)}
                                >
                                    {Object.entries(DATES).map(([label, value]) => (
                                        <option key={value} value={value}>
                                            {label}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div className="mb-3 flex-grow-1">
                                <label
                                    htmlFor="sonification-select"
                                    className="form-label fw-medium"
                                >
                                    Sonification
                                </label>
                                <select
                                    id="sonification-select"
                                    className="form-select"
                                    value={sonification}
                                    onChange={(e) => setSonification(e.target.value)}
                                >
                                    {Object.entries(SONIFICATION_TYPES).map(([label, value]) => (
                                        <option key={value} value={value}>
                                            {label}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>
                        <hr />
                    </div>
                </div>
            </div>
        </div>
    );
}
