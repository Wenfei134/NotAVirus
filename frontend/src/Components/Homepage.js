import React from "react";
import Hello from "./Hello";

function Homepage(){
    return( 
        <div id="homepage">
            <div class="right">
                <h1>The future of medicine at your fingertips</h1>
                <p>Submit a cough</p>
                <form action="./prediction" method="POST" enctype="multipart/form-data">
                    <input type="file" name="audiofile"/>
                    <input type="submit"/>
                </form>
            </div>
            <Hello/>
        </div>

    );

}

export default Homepage;