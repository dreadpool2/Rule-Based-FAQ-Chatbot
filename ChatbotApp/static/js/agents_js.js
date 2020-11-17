//list all queries
var storedId = -1;
var cached = null;



//Document ready function for checking queries to be deleted. Needs to be faster ~ 1000ms



function onQueryClicked(userid){
    
    var user = cached.User[userid.toString()];
    var ques = cached.Question[userid.toString()];

    $("#progress").css("visibility", "visible");
    $.get("/setOngoing", { sessionId: user }).done(function(data) {
        // add to delete table;
        if(data == "Failure"){
            alert("Query taken up by another person!");
            $( "#chatbox" ).load(window.location.href + " #chatbox" );
        }
        else if(data == "Success"){
            $("#queryDiv").css("visibility", "visible");
            document.getElementById("querySelec").innerHTML = ques;

            $("#chatbox").css("pointer-events", "none");
            document.getElementById("userName").value = user;
            $("#tdSubmit").css("pointer-events", "all");

            $("#mainBackground").css("opacity", "0.5");
        }
        else{
            alert("Query already solved!");
            $( "#chatbox" ).load(window.location.href + " #chatbox" );
        }

        $("#progress").css("visibility", "hidden");

    });

}

function createQueryMessage(query_id, question, status, id){
    //var element = '<p class="query">'+session_id+'<br>'+question+'<br>'+answer+'<br>'+status+'</p>';
    var element = '<p id = "'+id+'" class="query botText" onclick= "onQueryClicked(this.id)" style="cursor:pointer" >'+question+'</p>';
    $("#chatbox").prepend(element);
}

$("#submitAll").click(function() {
    //for now, whenever the agent sends a reply, the status updates to closed.
    // in future ongoing will be taken care of
    setAgentHelp($("#userName").val(), $("#answerInput").val(), "closed");
    $( "#chatbox" ).load(window.location.href + " #chatbox" );
    //enable clicking again
    $("#chatbox").css("pointer-events", "all");
    $("#tdSubmit").css("pointer-events", "none");
});

$("#dropAll").click(function() {
    $.get("/dropAll", function(data) {
        $( "#chatbox" ).load(window.location.href + " #chatbox" );
    });
});

//Document ready function for checking new queries.
$(document).ready(function(){
    setInterval(function(){
        $.get("/agentHelp", { msg: "rawText" }).done(function(data) {

            var query = JSON.parse(data);
            cached = query;
            //alert(query.User["1"]);
            var arr_length = Object.keys(query.User).length;
            var numItems = $('.query').length;

            var diff = (arr_length) - numItems;

            if(diff != 0){

                var starting_index = numItems;
                var i = 0;
                for(i = starting_index; i < arr_length; i++){
                    var ID = i.toString();
                    createQueryMessage(query.User[ID], query.Question[ID], query.Status[ID], $('.query').length);
                }
            } 
        });
    }, 3000);
});

function setAgentHelp(userName, answer, status){
    $.get("/agentSentHelp", { userName: userName, answer: answer, status: status }).done(function(data) {
        $("#mainBackground").css("opacity", "1");
        $("#queryDiv").css("visibility", "hidden");
    });
}