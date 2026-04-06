import { useState, useRef, useEffect } from "react";
import "./App.css";

const DATES = [
    "2026_03_03",
    "2026_03_04",
    "2026_03_05",
    "2026_03_06",
    "2026_03_07",
    "2026_03_08",
    "2026_03_09",
    "2026_03_10",
    "2026_03_11",
    "2026_03_12",
    "2026_03_13",
    "2026_03_14",
    "2026_03_15",
];

const SONIFICATION_TYPES = ["buscount"];

const BIN = "30s";

export default function App() {
    const [date, setDate] = useState(DATES[DATES.length - 1]);
    const [sonification, setSonification] = useState(SONIFICATION_TYPES[0]);
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
                <div className="col-8 d-flex align-items-center justify-content-center rounded">
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
                <div className="col-4 d-flex align-items-center">
                    <div className="card w-100 shadow-sm">
                        <div className="card-body">
                            <div className="mb-3">
                                <label htmlFor="date-select" className="form-label fw-medium">
                                    Date
                                </label>
                                <select
                                    id="date-select"
                                    className="form-select"
                                    value={date}
                                    onChange={(e) => setDate(e.target.value)}
                                >
                                    {DATES.map((d) => (
                                        <option key={d} value={d}>
                                            {d.replaceAll("_", "-")}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div className="mb-3">
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
                                    {SONIFICATION_TYPES.map((s) => (
                                        <option key={s} value={s}>
                                            {s}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <hr />
                            <p className="small text-muted mb-0">
                                Playing:{" "}
                                <code>
                                    {date.replaceAll("_", "-")} / {sonification}
                                </code>
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
