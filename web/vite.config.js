import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "fs";
import path from "path";

const rendersDir = path.resolve(__dirname, "../renders");

export default defineConfig({
    base: "/capmetro-sonification/",
    plugins: [
        react(),
        {
            name: "serve-renders",
            configureServer(server) {
                server.middlewares.use("/renders", (req, res, next) => {
                    const filePath = path.join(rendersDir, req.url);

                    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
                        const stats = fs.statSync(filePath);

                        // Crucial: Browsers need the length to build the seek bar
                        res.writeHead(200, {
                            "Content-Type": "video/mp4",
                            "Content-Length": stats.size,
                            "Accept-Ranges": "bytes", // Tells the browser it CAN ask for chunks
                        });

                        fs.createReadStream(filePath).pipe(res);
                    } else {
                        res.statusCode = 404;
                        res.end();
                    }
                });
            },
        },
    ],
});
