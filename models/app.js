const video = document.getElementById("video");

const startVideo = () => {
      navigator.getUserMedia(
        { video : {} },
        (stream) => (video.srcObject = stream),
        (err) => console.log(err)
      );
};


Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("./models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("./models"),

]).then(startVideo());

video.addEventListener("play",() => {

const canvas = faceapi.createCanvasFromMedia(video);
document.body.append(canvas);

const displaySize = {
  width: video.width,
  height: video.height,
};

  setInterval(async () => {
    const detections = await faceapi
    .detectAllFaces(video,new faceapi.tinyFaceDetectorOptions());
   
  const resizedDetection = faceapi.resizeResults(detections,displaySize);

  canvas.getContext("2d").clearRect(0,0,canvas.width,canvas.height);
  faceapi.draw.drawDetection(canvas,resizedDetection);

  },100);
});