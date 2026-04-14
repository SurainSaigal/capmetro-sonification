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

const SONIFICATION_TYPES = { "Single Sine Wave": "buscount", "Chord Progression": "avgspeed" };

const DESCRIPTIONS = {
    buscount:
        "This is the simplest example of sonification, where the number of buses on the road is represented by a single sine wave tone. The frequency of the sine wave increases as more buses are on the road, creating a rising pitch that reflects bus traffic volume. In order to truly feel the trends, a 24-hour period has been compressed to just over 2 minutes, where each frame of the video represents 30 seconds of real time (you can see the real time in the top right corner). \n\nThe accompanying animation is a visual representation to complement the sound you are hearing. In it, each bus is shown at it's current location, along with a trail representing the past 2 minutes of location data.",
    avgspeed:
        "This is a more complicated and artistic approach to sonification. Here, we have an underlying composition driving the chords, but the tempo, number of notes in the chord voicings, and pitch drift are all controlled by the bus data. \n\nYou can hear how there is a sparse, slow arrangement in the early morning times, when not much activity occurs. However, in the peak times of day, we hear a lush and fast arrangement. We also hear the pitch drift (detunement) of the chords increase during periods of high congestion, controlled by traffic speed.",
};

const BIN = "30s";

export default function App() {
    const [date, setDate] = useState(DATES["Wednesday, March 4"]);
    const [sonification, setSonification] = useState(SONIFICATION_TYPES["Single Sine Wave"]);
    const [notFound, setNotFound] = useState(false);
    const videoRef = useRef(null);

    const videoSrc = `${import.meta.env.BASE_URL}renders/${date}_${sonification}_${BIN}.mp4`;

    // Reset not-found state and reload video when selection changes
    useEffect(() => {
        setNotFound(false);
        if (videoRef.current) {
            videoRef.current.load();
        }
    }, [videoSrc]);

    return (
        <div className="container-fluid vh-100 d-flex flex-column py-3 bg-dark text-light overflow-hidden">
            <div className="row flex-grow-1 g-3 align-items-stretch">
                {/* Video panel */}
                <div className="col-7 d-flex align-items-center justify-content-center rounded">
                    <div className="video-frame">
                        {notFound ? (
                            <div className="text-center text-secondary">
                                <p className="fs-5 mb-1">Not yet rendered</p>
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
                        <h4 className="mb-3 text-center fw-semibold">
                            CapMetro Sonification Gallery{" "}
                            <em className="text-secondary">by Surain Saigal</em>
                        </h4>
                        <p className="">
                            This gallery shows various methods of "sonifying" the data of Austin's
                            CapMetro bus system. Data Sonification is the process of transforming
                            raw data into sound to analyze and interpret information through audio.
                            In this case, I have taken CapMetro GPS data, available freely through
                            the Texas Open Data Portal, and demonstrated multiple ways to sonify
                            this data. All visuals and audio have been generated programmatically.
                            Feel free to adjust the selectors below to explore.
                        </p>
                        <hr />
                        <div className="d-flex gap-3 align-items-start">
                            <div className="mb-3 flex-grow-1">
                                <label htmlFor="date-select" className="form-label fw-medium">
                                    Date
                                </label>
                                <select
                                    id="date-select"
                                    className="form-select btn btn-secondary"
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
                                    Sonification Type
                                </label>
                                <select
                                    id="sonification-select"
                                    className="form-select btn btn-secondary"
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
                        <p style={{ whiteSpace: "pre-line" }} className="fs-6">
                            {DESCRIPTIONS[sonification]}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
