import React, {useEffect, useState, useReducer} from "react";
import Hello from "./Components/Hello";
import Typography from "@material-ui/core/Typography";
import Button from "@material-ui/core/Button";
import IconButton from '@material-ui/core/IconButton';
import Paper from "@material-ui/core/Paper";
import Container from "@material-ui/core/Container";
import AudiotrackIcon from '@material-ui/icons/Audiotrack';
import LinearProgressionWithLabel from "./Components/LinProgWithLabel";
import { makeStyles } from '@material-ui/core/styles';
import logo from './logo.svg';
import Results from './Components/Results';
import './styles.css';

const useStyles = makeStyles((theme) => ({
  mainContainer: {
    paddingTop: "10%",
  },
  contentContainer: {
    display: "flex",
    alignItems: "center",
    flexDirection: "column",
    "& > #appDiv" : {
      marginTop: "10px"
    } 
  },
  input: {
    display: "none",
  }
}));
function App() {
  const classes = useStyles();
  const [stage, setStage] = useState(0); // 0 FOR NO UPLOAD, 1 FOR UPLOAD, 2 FOR RESULT
  const [COVIDFree, setCOVID] = useState(false);
  const [progress, setProgress] = useState(10);
  const [confidence, setConfidence] = useState(0);
  const [result, setResult] = useState("");

  console.log(COVIDFree)
  return (
  <React.Fragment>
    <div class="header">
        <a href="./SignIn">Sign In</a>
        <a href="./Register">Register</a>
    </div>
    <div id="homepage">


      <Container className={classes.mainContainer}>
          <form action="./prediction" method="POST" enctype="multipart/form-data">
          <Paper elevation={3} className={classes.contentContainer}>
            <Typography component="div" variant="h4" id="appDiv">Submit Your Cough</Typography> 
            <div id="appDiv"> 

            {/* We need a for loop to constantly fetch the back end for progress */}
            <input type="file" onChange={(e) => { 
              setStage(1) 
              setProgress(10);
              let name = e.target.name;
              let value = e.target.files[0];
              let formData = new FormData();
              formData.append(name, value);
              fetch("/prediction", {
                method: "POST",
                body: formData,
              }).then((res) => res.json())
              .then((result) => {
                  setResult(result["result"]);
                  setConfidence(result["confidence"])
              }).catch((err) => alert(err));
              const timer = setInterval(() => {
                let currProgress = 0;
                
                setProgress((prevProgress) => { 
                  if (prevProgress < 100)
                    currProgress = prevProgress + 10
                  else return prevProgress 
                  return currProgress
                });
                if (currProgress === 100){ 
                  setCOVID(true);
                  setStage(2);
                  clearInterval(timer);
                }
              }, 800);
              return () => {
                clearInterval(timer);
              };
            } } className={classes.input} name="audiofile" id="contained-button-file" accept=".wav"/>
            <label htmlFor="contained-button-file">
              <Button variant="contained" color="primary" component="span">Upload Audio</Button>  
            </label>          
            <IconButton color="primary" aria-label="upload audiofile" component="span">
              <AudiotrackIcon />
            </IconButton>
            </div>                    
            { (stage === 1) && <LinearProgressionWithLabel value={progress}/> }   
            { (stage === 2) && <Results COVIDFree={(result === "positive") ? false : true} confidence={confidence}/>}
            </Paper>
          </form>
          
      </Container>
    </div>

  </React.Fragment>
  );
}

export default App;

