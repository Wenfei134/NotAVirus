import React from "react";
import Hello from "./Components/Hello";
import logo from './logo.svg';
import './styles.css';

function App() {
  return (
  <React.Fragment>
    <div class="header">
        <a href="./SignIn">Sign In</a>
        <a href="./Register">Register</a>
    </div>

    <div id="homepage">
        <div class="right">
            <h1>The future of medicine at your fingertips</h1>
            <p>Submit a cough</p>
            <form action="./prediction" method="POST" enctype="multipart/form-data">
                <input type="file" name="audiofile"/>
                <input type="submit"/>
            </form>
        </div>
        
    </div>
    <Hello/>
  </React.Fragment>
  );
}

export default App;
