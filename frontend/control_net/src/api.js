export const processImage = async (formData) => {
    try {
      const response = await fetch('http://localhost:8000/test-model/process', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return await response.json();
    } catch (error) {
      console.error('Error processing image:', error);
      return null;
    }
  };