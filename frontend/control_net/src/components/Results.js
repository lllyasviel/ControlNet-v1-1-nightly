import React from 'react';
import styled from 'styled-components';

const ResultsContainer = styled.div`
  margin-top: 20px;
`;

// const ResultImage = styled.img`
//   max-width: 100%;
//   margin-bottom: 10px;
// `;

const ResultImage = styled.img`
  width: 500px;
  height: 400px;
  margin: 0 auto 20px;
  display: flex;
  justify-content: center;
  align-items: center;
  overflow: hidden;
  background-color: #f0f0f0;
  border-radius: 10px;
`;


function Results({ results }) {
  return (
    <ResultsContainer>
      <h2>처리 결과</h2>
      <h3>처리 시간: {results.process_time}초</h3>
      <h4>입력 원본이미지</h4>
      <ResultImage src={`http://localhost:8000/image/?file_path=${results.input_image}`} alt={`입력 이미지`} />
      {results.output_paths.map((path, index) => (
        <div key={index}>
          <h4>사진{index + 1}</h4>
          <p>FID 점수(낮을수록 좋음. 10 이하가 이상적): <b>{results.score_fid[index]}</b></p>
          <p>SSIM 점수(1에 가까울수록 좋음. 0.9 이상이 이상적): <b>{results.score_ssim[index]}</b></p>
          <ResultImage src={`http://localhost:8000/image/?file_path=${path}`} alt={`결과 이미지 ${index + 1}`} />
        </div>
      ))}
    </ResultsContainer>
  );
}

export default Results;