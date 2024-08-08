// src/Game.js
import React, { useState, useEffect } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import "./Game.css";

const Game = () => {
  // 기존 App 컴포넌트의 내용
  localStorage.removeItem("pixabayGameScore");
  const [sketchUrl, setSketchUrl] = useState(null);
  const [originalUrl, setOriginalUrl] = useState(null);
  const [keywords, setKeywords] = useState([]);
  const [selectedKeyword, setSelectedKeyword] = useState("");
  const [message, setMessage] = useState("");
  const [score, setScore] = useState(0);
  const [difficulty, setDifficulty] = useState("medium");
  const [hint, setHint] = useState("");
  const [showOriginal, setShowOriginal] = useState(false);

  useEffect(() => {
    const savedScore = localStorage.getItem("pixabayGameScore");
    if (savedScore) setScore(parseInt(savedScore));
  }, []);

  const startGame = async () => {
    setMessage("");
    setHint("");
    setShowOriginal(false);
    setSketchUrl(null);
    setOriginalUrl(null);

    try {
      const response = await axios.post(
        `http://localhost:8000/sketch-image/start_game`
      );

      const sketchResponse = await axios.get(
        "http://localhost:8000/image/",
        {
          params: {
            file_path: response.data.sketch_url,
          },
        }
      );
      

      const originalResponse = await axios.get(
        "http://localhost:8000/image/",
        {
          params: {
            file_path: response.data.original_url,
          },
        }
      );
      setSketchUrl(sketchResponse.request.responseURL);
      setOriginalUrl(originalResponse.request.responseURL);
      setKeywords(response.data.keywords);
      setSelectedKeyword("");
    } catch (error) {
      console.error("게임 시작 오류:", error);
      setSketchUrl(null);
    }
  };

  const handleKeywordChange = (e) => {
    setSelectedKeyword(e.target.value);
  };

  const submitGuess = () => {
    if (keywords.includes(selectedKeyword)) {
      const isCorrect =
        selectedKeyword.toLowerCase() === sketchUrl.split('%2F').pop().split('_')[0];
      if (isCorrect) {
        setMessage("정답입니다! 원본 이미지를 확인하세요.");
        setScore(score + 10);
        localStorage.setItem("pixabayGameScore", score + 10);
        setShowOriginal(true);
      } else {
        setMessage("틀렸습니다. 다시 시도해보세요!");
        setScore(Math.max(0, score - 5));
        localStorage.setItem("pixabayGameScore", Math.max(0, score - 5));
      }
    } else {
      setMessage("유효한 키워드를 선택해주세요.");
    }
  };

  const getHint = () => {
    const correctKeyword = sketchUrl.split('%2F').pop().split('_')[0];

    setHint(`정답은 "${correctKeyword[0]}"로 시작합니다`);
    setScore(Math.max(0, score - 2));
    localStorage.setItem("pixabayGameScore", Math.max(0, score - 2));
  };

  return (
    <div className="app-container">
      <h1>이미지 키워드 맞추기</h1>
      <div className="score">점수: {score}</div>
      {!sketchUrl ? (
        <div className="start-screen">
          <select
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
            className="difficulty-select"
          >
            <option value="easy">쉬움</option>
            <option value="medium">보통</option>
            <option value="hard">어려움</option>
          </select>
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={startGame}
            className="start-button"
          >
            게임 시작
          </motion.button>
        </div>
      ) : (
        <div className="game-screen">
          <h2>이 이미지의 키워드를 맞춰보세요:</h2>
          <div className="image-container">
            <motion.img
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
              src={sketchUrl}
              alt="Pixabay 스케치"
              className="sketch-image"
            />
          </div>
          <div className="guess-container">
            <select
              value={selectedKeyword}
              onChange={handleKeywordChange}
              className="keyword-select"
            >
              <option value="" disabled>
                키워드 선택
              </option>
              {keywords.map((keyword) => (
                <option key={keyword} value={keyword}>
                  {keyword}
                </option>
              ))}
            </select>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={submitGuess}
              className="submit-button"
            >
              제출
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={getHint}
              className="hint-button"
            >
              힌트 보기 (-2점)
            </motion.button>
          </div>
          {message && <p className="message">{message}</p>}
          {hint && <p className="hint">{hint}</p>}
          {showOriginal && (
            <div className="original-image-container">
              <h3>원본 이미지:</h3>
              <div className="image-container">
                <motion.img
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5 }}
                  src={originalUrl}
                  alt="Pixabay 원본"
                  className="original-image"
                />
              </div>
            </div>
          )}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={startGame}
            className="next-button"
          >
            다음 이미지
          </motion.button>
        </div>
      )}
    </div>
  );
};

export default Game;
