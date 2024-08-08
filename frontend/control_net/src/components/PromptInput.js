import React from 'react';
import styled from 'styled-components';

const Input = styled.input`
  width: 100%;
  padding: 10px;
  margin-bottom: 20px;
  border: 1px solid #ccc;
  border-radius: 4px;
`;

function PromptInput({ prompt, setPrompt }) {
  return (
    <Input 
      type="text" 
      value={prompt}
      onChange={(e) => setPrompt(e.target.value)}
      placeholder="프롬프트를 입력하세요"
    />
  );
}

export default PromptInput;