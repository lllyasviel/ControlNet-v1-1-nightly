import React, { useState } from 'react';
import styled from 'styled-components';
import ImageUpload from './ImageUpload';
import PromptInput from './PromptInput';
import AdvancedOptions from './AdvancedOptions';
import Results from './Results';
import { processImage } from '../api';

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: Arial, sans-serif;
`;

const Title = styled.h1`
  text-align: center;
  color: #333;
`;

const Button = styled.button`
  background-color: #4285f4;
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  margin-top: 20px;
  width: 100%;
  &:disabled {
    background-color: #a3c9f7;
    cursor: not-allowed;
  }
`;

const LoadingBar = styled.div`
  width: 100%;
  height: 4px;
  background-color: #4285f4;
  animation: loading 2s infinite;

  @keyframes loading {
    0% { width: 0; }
    50% { width: 100%; }
    100% { width: 0; }
  }
`;

function ImageProcessor() {
  const [image, setImage] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [options, setOptions] = useState({
    num_samples: 1,
    image_resolution: 768,
    low_threshold: 100,
    high_threshold: 200,
    num_steps: 20,
    guidance_scale: 9,
    seed: 0,
    randomize_seed: true,
    a_prompt: 'best quality, extremely detailed',
    n_prompt: 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
  });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true); // 로딩 시작
    setResults(null);
    const formData = new FormData();
    formData.append('input_image', image);
    formData.append('prompt', prompt);

    Object.keys(options).forEach(key => {
      formData.append(key, options[key]);
    });

    try {
      const response = await processImage(formData);
      setResults(response);
    } catch (error) {
      console.error("Error processing image:", error);
    } finally {
      setLoading(false); // 로딩 종료
    }
  };

  return (
    <Container>
      <Title>모델 테스트 페이지</Title>
      <ImageUpload setImage={setImage} />
      <PromptInput prompt={prompt} setPrompt={setPrompt} />
      <AdvancedOptions options={options} setOptions={setOptions} />
      {loading && <LoadingBar />}
      <Button onClick={handleSubmit} disabled={loading}>실행</Button>
      {results && <Results results={results} />}
    </Container>
  );
}

export default ImageProcessor;
