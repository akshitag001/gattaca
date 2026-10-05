"""OpenCV 5 pipeline stages: detect -> roi -> rectify -> enhance -> extract -> quality
(-> deform -> match for the measurement study). Each stage is a pure function and
accepts an optional `debug` sink that dumps its intermediate images."""
