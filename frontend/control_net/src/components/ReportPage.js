import React, { useEffect, useState } from 'react';

function ReportPage() {
  const [reports, setReports] = useState([]);

  useEffect(() => {
    const savedResults = JSON.parse(localStorage.getItem('results') || '[]');
    setReports(savedResults);
  }, []);

  return (
    <div className="report-page">
      <h1>처리 결과 리포트</h1>
      {reports.map((report, index) => (
        <div key={index} className="report-item">
          <h2>결과 #{index + 1}</h2>
          <p>처리 시간: {report.process_time}초</p>
          <p>FID 점수: {report.fid_score}</p>
          <p>FSNR 점수: {report.fsnr_score}</p>
          <p>SSIM 점수: {report.ssim_score}</p>
          {report.output_paths.map((path, i) => (
            <img key={i} src={path} alt={`결과 이미지 ${i + 1}`} />
          ))}
        </div>
      ))}
    </div>
  );
}

export default ReportPage;