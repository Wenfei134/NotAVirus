import React from 'react'

function PlayAudio(audiofile){
    return(
        <audio controls>
            <source src={audiofile} type='audio/wav'/>
            Your browser does not support the audio tag.
        </audio>
    ); 
}

export default PlayAudio;
