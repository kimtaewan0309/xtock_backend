const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
require("dotenv").config();

const simulationRoutes = require("./simulation/routes/simulationRoutes");

const app = express();

app.use(cors());
app.use(express.json());

mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log("MongoDB 연결 성공"))
  .catch((err) => console.error("MongoDB 연결 실패:", err));

app.use("/api/simulation", simulationRoutes);

module.exports = app;