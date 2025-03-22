const express = require("express");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const mammoth = require("mammoth");
const fs = require("fs-extra");
const cors = require("cors");
const axios = require("axios");

const app = express();
const port = 5000;
const AI_MODEL_URL = "http://127.0.0.1:5001/process-resume"; // Flask AI Model URL

app.use(cors());
app.use(express.json());

// Multer setup for multiple file uploads
const upload = multer({ dest: "uploads/" });

app.post("/api/extract-text", upload.array("files"), async (req, res) => {
  try {
    let extractedResumes = [];

    // Process each uploaded file
    for (let file of req.files) {
      const filePath = file.path;
      const fileType = file.mimetype;
      let extractedText = "";

      if (fileType === "application/pdf") {
        const data = await pdfParse(fs.readFileSync(filePath));
        extractedText = data.text;
      } else if (
        fileType ===
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
      ) {
        const data = await mammoth.extractRawText({ path: filePath });
        extractedText = data.value;
      } else if (fileType === "text/plain") {
        extractedText = fs.readFileSync(filePath, "utf8");
      } else {
        return res.status(400).json({ error: "Unsupported file format" });
      }

      extractedResumes.push({
        id: `resume-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        filename: file.originalname,
        text: extractedText,
      });

      fs.removeSync(filePath);
    }

    // Send extracted text to AI model
    // let aiResponse = await axios.post(AI_MODEL_URL, { resumes: extractedResumes });

    // Send response to frontend
    res.json({
      extracted_resumes: extractedResumes,
      // ai_parsed_resumes: aiResponse.data,
    });
    console.log("hi");
    console.log(extractedResumes);
    console.log("hello");
    // console.log(aiResponse.data);
  } catch (error) {
    console.error("Error processing file:", error);
    res.status(500).json({ error: "Error extracting text" });
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
