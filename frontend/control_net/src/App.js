import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import styled from 'styled-components';
import Game from './Game';
import ImageProcessor from './components/ImageProcessor';
import ReportPage from './components/ReportPage';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background-color: #f0f4f8;
  font-family: 'Arial', sans-serif;
`;

const Title = styled.h1`
  color: #2c3e50;
  font-size: 2.5rem;
  margin-bottom: 2rem;
`;

const NavContainer = styled.nav`
  display: flex;
  gap: 1rem;
`;

const StyledLink = styled(Link)`
  text-decoration: none;
`;

const Button = styled.button`
  padding: 0.8rem 1.5rem;
  font-size: 1rem;
  color: #fff;
  background-color: #3498db;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s ease;

  &:hover {
    background-color: #2980b9;
  }
`;

const Home = () => (
  <Container>
    <Title>ProtoType ControlNet</Title>
    <NavContainer>
      <StyledLink to="/image-processor">
        <Button>모델 테스트 페이지</Button>
      </StyledLink>
      <StyledLink to="/report">
        <Button>처리 결과 보기</Button>
      </StyledLink>
      <StyledLink to="/game">
        <Button>간단한 게임</Button>
      </StyledLink>
    </NavContainer>
  </Container>
);

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/game" element={<Game />} />
        <Route path="/image-processor" element={<ImageProcessor />} />
        <Route path="/report" element={<ReportPage />} />
      </Routes>
    </Router>
  );
};

export default App;