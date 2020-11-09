import React, {useState, useEffect} from "react";
import Typography from "@material-ui/core/Typography";
import { makeStyles } from "@material-ui/core/styles";
import EmojiEmotionsIcon from '@material-ui/icons/EmojiEmotions';
import InsertEmoticonIcon from '@material-ui/icons/InsertEmoticon';
import {
    FacebookShareButton,
    LinkedinShareButton,
    TwitterShareButton,
    FacebookIcon,
    LinkedinIcon,
    TwitterIcon,
} from 'react-share';

const useStyles = makeStyles((theme) => ({
    resultsContainer : {
        width: "80%",
        justifyContent: "center",
    },
    resultsText: {
        fontWeight: 800
    }
}))

var joy = "Congrats! You are COVIDFree";
var death = "You have COVID YOU WILL DIE"; 
function Results({ COVIDFree, confidence }){
    const classes = useStyles();
    const [result, setResult] = useState("");
    const chance = " with a " + confidence.toFixed(2) + "% chance"; 
    useEffect(() => {
        setResult(() => (COVIDFree) && joy || death);
    }, [COVIDFree]);
    
    return (
    <div className={classes.resultsContainer}>
        <Typography className={classes.resultsText} variant="h3" component="div">{result + chance}</Typography> 
         {(COVIDFree) && <EmojiEmotionsIcon/>
          || <InsertEmoticonIcon/>}
        <div>
        <FacebookShareButton url="facebook.com" quote={result + chance} hashtag="mycovidresults"><FacebookIcon size={32} round={true}></FacebookIcon></FacebookShareButton>
        <LinkedinShareButton url="linkedin.ca" title={"My Covid Results"} summary={result + chance} source={"linkedin.ca"}><LinkedinIcon size={32} round={true}></LinkedinIcon></LinkedinShareButton>
        <TwitterShareButton url="twitter.com" title={"My Covid Results"} via={"myself"} hashtags={["mycovidresults", "awesome"]}><TwitterIcon size={32} round={true}></TwitterIcon></TwitterShareButton>
        </div> 
    </div>
    );
}

export default Results;