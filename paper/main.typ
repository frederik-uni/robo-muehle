#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Robot Playing Nine Men's Morris],
  paper-size: "a4",
  abstract: [This work presents a robotic system capable of autonomously playing Nine Men’s Morris by integrating computer vision and strategic decision-making. The system employs a robotic manipulator to physically move pieces on the board while leveraging YOLO-based visual perception to detect the current board state in real time. A custom-trained model evaluates the game state and selects optimal moves, enabling the robot to play strategically against human or simulated opponents. We detail the design of the perception pipeline, the interfacing of the robot arm with the board, and the training of the game model, emphasizing how these components interact to achieve seamless gameplay.],
  authors: (
    (
      name: "Eugen Ganscha",
      organization: [Studiengang Informatik],
      location: [Matrikelnummer: 00514322],
      email: "eganscha@hof-university.de"
    ),
    (
      name: "Frederik Schwarz",
      organization: [Studiengang Informatik],
      location: [Matrikelnummer: 00515923],
      email: "frederik.schwarz@hof-university.de"
    ),
    (
      name: "Karla Schramm",
      organization: [Studiengang Informatik],
      location: [Matrikelnummer: ],
      email: "karla.schramm@hof-university.de"
    ),
    (
      name: "Kitti Kern",
      organization: [Studiengang Medieninformatik],
      location: [Matrikelnummer: 00646623],
      email: "kitti.kern@hof-university.de"
    ),
    (
      name: "Marie Müller",
      organization: [Studiengang Medieninformatik],
      location: [Matrikelnummer: 00286622],
      email: "marie.mueller@hof-university.de"
    ),
  ),
  bibliography: bibliography("refs.bib"),
)
#include "intro.typ"
#include "detect.typ"
#include "ai.typ"
#include "movement.typ"
