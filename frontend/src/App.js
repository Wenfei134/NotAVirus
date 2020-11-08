import React from "react";
import Homepage from "./Components/Homepage";
import logo from './logo.svg';
import './styles.css';

function App() {
  return (
  <React.Fragment>
    <div class="header">
        <a href="./SignIn">Sign In</a>
        <a href="./Register">Register</a>
    </div>
    <Homepage />
  </React.Fragment>
  );
}

export default App;
