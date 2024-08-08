import React from 'react';
import styled from 'styled-components';

const UploadContainer = styled.div`
  border: 2px dashed #ccc;
  border-radius: 4px;
  padding: 20px;
  text-align: center;
  margin-bottom: 20px;
`;

function ImageUpload({ setImage }) {
  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
    }
  };

  return (
    <UploadContainer>
      <input type="file" onChange={handleImageChange} accept="image/*" />
      <p>이미지를 끌어다 놓거나 클릭하여 업로드하세요</p>
    </UploadContainer>
  );
}

export default ImageUpload;