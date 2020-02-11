
//GUI COMPONENTS

    // BASIC WINDOW
    function createWindow(myWindow){
      var x = myWindow.x;
      var y = myWindow.y;
      var width = myWindow.width;
      var height = myWindow.height;
      var color  = myWindow.color;
      var title  = myWindow.title;
      var textColor = myWindow.textColor;
  
      context.fillStyle = color;
      context.fillRect(x,y,width,height);
      
      //Add Deletion button
      dButton = {x:x+width-25, y:y, width:25, height:25};
      deleteWindow(dButton,myWindow);
      //Add Hide button
      hButton = {x:x+width-50, y:y, width:25, height:25};
      hideWindow(hButton,myWindow);
  
      //Title
      context.fillStyle = textColor;
      var font_size = width/10;
      context.font = font_size.toString() + "px Arial";
      context.textAlign = "center";
      context.fillText(title,x+width/2,y+height/6);
      
      //Title & Content Division Line
      drawline(x,y+height/5,x+width,y+height/5, "ivory");
    }

     function createButton(myButton){
      var x = myButton.x;
      var y = myButton.y;
      var width = myButton.width;
      var height = myButton.height;
      var color  = myButton.color;
      var text  = myButton.text;
      var textColor = myButton.textColor;
  
      context.fillStyle = color;
      context.fillRect(x,y,width,height);

      //Add Alert Event
      AddAlertEvent(myButton);

      //Add Deletion button
      dButton = {x:x+width-25, y:y, width:25, height:25};
      deleteWindow(dButton,myButton);
  
      //Title
      context.fillStyle = textColor;
      var font_size = width/10;
      context.font = font_size.toString() + "px Arial";
      context.textAlign = "center";
      context.fillText(text,x+width/2,y+height/2);
    }
    
    function createTextBox(myTextBox){
      var x = myTextBox.x;
      var y = myTextBox.y;
      var width = myTextBox.width;
      var height = myTextBox.height;
      var color  = myTextBox.color;
      var text  = myTextBox.text;
      var textColor = myTextBox.textColor;
      var textSize  = myTextBox.textSize;
      var exit = myTextBox.exit;
     
      context.fillStyle = color;
      context.fillRect(x,y,width,height);
      
      //Add Deletion button
      if(exit == true){
        dButton = {x:x+width-25, y:y, width:25, height:25};
        deleteWindow(dButton,myTextBox);
      }
      //Title
      context.fillStyle = textColor;
      var font_size = width/10;
      context.font =  textSize.toString() + "px Arial";
      context.textAlign = "center";
      context.fillText(text,x+width/2,y+height/2);
    }

    function createMenu(myMenu){
      var x = myMenu.x;
      var y = myMenu.y;
      var width = myMenu.width;
      var height = myMenu.height;
      var color  = myMenu.color;
      var text  = myMenu.text;
      var textColor = myMenu.textColor;
      var textSize  = myMenu.textSize;
      var menuList = myMenu.menuList;
     
      context.fillStyle = color;
      context.fillRect(x,y,width,height);
      
      //Add Deletion button
      dButton = {x:x+width-25, y:y, width:25, height:25};
      deleteWindow(dButton,myMenu);
      
      //Add Menu button
      mButton = {x:x, y:y, width:25, height:25};
      menuWindow(mButton,myMenu);
  
      //Title
      context.fillStyle = textColor;
      var font_size = width/10;
      context.font =  textSize.toString() + "px Arial";
      context.textAlign = "center";
      context.fillText(text,x+width/2,y+height/2);
    }






//BASIC COMPONENTS
  function getOffset(el) {
    const rect = el.getBoundingClientRect();
    return {
      left: rect.left + window.scrollX,
      top: rect.top + window.scrollY
    };
  }

  function getMousePos(canvas, event) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
  }

  function drawline(x1,y1,x2,y2, color){
    context.beginPath();
    context.moveTo(x1,y1);
    context.lineTo(x2,y2);
    context.strokeStyle = color;
    context.stroke();
    context.closePath();
  }




  //DELETION BUTTON HANDLER
  function deleteWindow(dButton,rect){
    context.fillStyle = "lightcoral";
    context.fillRect(dButton.x,dButton.y,dButton.width,dButton.height);
    AddDeletionEvent(dButton, rect);
    drawline(dButton.x+3,dButton.y+3,dButton.x+dButton.width-3,dButton.y+dButton.height-3, "white");
    drawline(dButton.x+dButton.width-3,dButton.y+3,dButton.x+3,dButton.y+dButton.height-3, "white");
  }
  //HIDE BUTTON HANDLER
  function hideWindow(hButton,rect){
    context.fillStyle = "lightgrey";
    context.fillRect(hButton.x,hButton.y,hButton.width,hButton.height);
    AddHideEvent(hButton, rect);
    drawline(hButton.x,hButton.y+hButton.height/2,hButton.x+hButton.width,hButton.y+hButton.height/2, "white");
  }
  //HIDE BUTTON HANDLER
  function menuWindow(mButton,rect){
    context.fillStyle = "lightgrey";
    context.fillRect(mButton.x,mButton.y,mButton.width,mButton.height);
    AddMenuEvent(mButton, rect);
    drawline(mButton.x,mButton.y+mButton.height/2,mButton.x+mButton.width,mButton.y+mButton.height/2, "white");
    drawline(mButton.x + mButton.width/2,mButton.y+mButton.height,mButton.x+mButton.width/2,mButton.y, "white");
  }

  function isInside(pos, rect){
    return pos.x > rect.x && pos.x < rect.x+rect.width && pos.y < rect.y+rect.height && pos.y > rect.y
  }

  //if dButton is clicked, remove rectangle
  function AddDeletionEvent(dButton,rect){
    canvas.addEventListener('click', function(evt) {
      var mousePos = getMousePos(canvas, evt);
      if (isInside(mousePos,dButton)) {
        context.clearRect(rect.x-1,rect.y-1,rect.width+2,rect.height+2);
      }
    }, false);
  }

  //if hButton is clicked, remove rectangle
  function AddHideEvent(dButton,rect){
    canvas.addEventListener('click', function(evt) {
      var mousePos = getMousePos(canvas, evt);
      if (isInside(mousePos,dButton)) {
        //context.clearRect(rect.x-1,rect.y-1,rect.width+2,rect.height+2);
      }
    }, false);
  }

  //if Button Object is clicked, give alert announcement to user
  function AddAlertEvent(Button){
    canvas.addEventListener('click', function(evt) {
      var mousePos = getMousePos(canvas, evt);
      if (isInside(mousePos,Button)) {
        alert("This button is clicked");
      }
    }, false);
  }

  //if Button is clicked, show menus
  function AddMenuEvent(Button,rect){
    var clicked = false;
    canvas.addEventListener('click', function(evt) {
      var mousePos = getMousePos(canvas, evt);
      if (isInside(mousePos,Button)) {
        if(clicked == false){
          for(var i =0; i<rect.menuList.length; i++){
            var menu = {
              x:rect.x, y:rect.y+(rect.height+1)*(i+1), width:rect.width/2, height:rect.height,
              color:rect.color, 
              text:rect.menuList[i], textColor:rect.textColor, textSize:rect.textSize, exit:false};
            createTextBox(menu);
          }
          clicked = true;
        }
        else{
          context.clearRect(rect.x,rect.y+(rect.height+1),rect.width/2+1,rect.height*(rect.menuList.length+1));
          clicked = false;
        }
      }
    }, false);
  }

