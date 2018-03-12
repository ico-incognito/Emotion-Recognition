var video = document.querySelector("#videoElement");
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
var count = 0;

//Data points
var dps = [];
var dps0 = [];
var dps1 = [];
var dps2 = [];
var dps3 = [];
var dps4 = [];
var dps5 = [];
var dps6 = [];
var xVal = 0;

//Defining scales of axes
var yViewportMin = 0
var yViewportMax = 1
var yMin = 0
var yMax = 1

var xViewportMin = 0
var xViewportMax = 20
var xMin = 0
var xMax = 20

//Number of dataPoints visible at any point
var dataLength = 40;

//Charts
var chart0 = new CanvasJS.Chart("chartContainer0", {
	axisY: {
		viewportMinimum : yViewportMin,
		viewportMaximum : yViewportMax,
		minimum : yMin,
		maximum : yMax,
		includeZero: true
	},
	axisX: {
		viewportMinimum : xViewportMin,
		viewportMaximum : xViewportMax,
		minimum : xMin,
		maximum : xMax,
		includeZero: true
	},
	data: [{
		type: "line",
		dataPoints: dps0
	}]
});

var chart1 = new CanvasJS.Chart("chartContainer1", {
	axisY: {
		viewportMinimum : yViewportMin,
		viewportMaximum : yViewportMax,
		minimum : yMin,
		maximum : yMax,
		includeZero: true
	},
	axisX: {
		viewportMinimum : xViewportMin,
		viewportMaximum : xViewportMax,
		minimum : xMin,
		maximum : xMax,
		includeZero: true
	},
	data: [{
		type: "line",
		dataPoints: dps1
	}]
});

var chart2 = new CanvasJS.Chart("chartContainer2", {
	axisY: {
		viewportMinimum : yViewportMin,
		viewportMaximum : yViewportMax,
		minimum : yMin,
		maximum : yMax,
		includeZero: true
	},
	axisX: {
		viewportMinimum : xViewportMin,
		viewportMaximum : xViewportMax,
		minimum : xMin,
		maximum : xMax,
		includeZero: true
	},
	data: [{
		type: "line",
		dataPoints: dps2
	}]
});

var chart3 = new CanvasJS.Chart("chartContainer3", {
	axisY: {
		viewportMinimum : yViewportMin,
		viewportMaximum : yViewportMax,
		minimum : yMin,
		maximum : yMax,
		includeZero: true
	},
	axisX: {
		viewportMinimum : xViewportMin,
		viewportMaximum : xViewportMax,
		minimum : xMin,
		maximum : xMax,
		includeZero: true
	},
	data: [{
		type: "line",
		dataPoints: dps3
	}]
});

var chart4 = new CanvasJS.Chart("chartContainer4", {
	axisY: {
		viewportMinimum : yViewportMin,
		viewportMaximum : yViewportMax,
		minimum : yMin,
		maximum : yMax,
		includeZero: true
	},
	axisX: {
		viewportMinimum : xViewportMin,
		viewportMaximum : xViewportMax,
		minimum : xMin,
		maximum : xMax,
		includeZero: true
	},
	data: [{
		type: "line",
		dataPoints: dps4
	}]
});

var chart5 = new CanvasJS.Chart("chartContainer5", {
	axisY: {
		viewportMinimum : yViewportMin,
		viewportMaximum : yViewportMax,
		minimum : yMin,
		maximum : yMax,
		includeZero: true
	},
	axisX: {
		viewportMinimum : xViewportMin,
		viewportMaximum : xViewportMax,
		minimum : xMin,
		maximum : xMax,
		includeZero: true
	},
	data: [{
		type: "line",
		dataPoints: dps5
	}]
});

var chart6 = new CanvasJS.Chart("chartContainer6", {
	axisY: {
		viewportMinimum : yViewportMin,
		viewportMaximum : yViewportMax,
		minimum : yMin,
		maximum : yMax,
		includeZero: true
	},
	axisX: {
		viewportMinimum : xViewportMin,
		viewportMaximum : xViewportMax,
		minimum : xMin,
		maximum : xMax,
		includeZero: true
	},
	data: [{
		type: "line",
		dataPoints: dps6
	}]
});

//Handle webcam 
navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia || navigator.oGetUserMedia;

if (navigator.getUserMedia) {       
	navigator.getUserMedia({video: true}, handleVideo, videoError);
}

function handleVideo(stream) {
	video.src = window.URL.createObjectURL(stream);
}

function videoError(e) {
	//Error
}

//Update charts
var updateChart = function (chart, dps, emotion) {
		dps.push({
			x: xVal,
			y: emotion
		});
	if (dps.length > dataLength) {
		dps.shift();
	}
	chart.render();
};

document.getElementById("snap").addEventListener("click", function() {
	//Execute every half second
	var interval = setInterval(function(){
		//Stopping condition
		if(count > 40)
			clearInternal(interval);
		else
		{
			count += 1
			var xhttp = new XMLHttpRequest();

			//Successful response received
			xhttp.onreadystatechange = function() {
				if (this.readyState == 4 && this.status == 200)
				{
					var data = JSON.parse(this.response);
					console.log(data)

					//Update charts
					updateChart(chart0, dps0, data.anger);
					updateChart(chart1, dps1, data.disgust * 1.5);
					updateChart(chart2, dps2, data.fear * 1.5);
					updateChart(chart3, dps3, data.happy);
					updateChart(chart4, dps4, data.sad);
					updateChart(chart5, dps5, data.surprise * 1.5);
					updateChart(chart6, dps6, data.neutral * 0.8);

					//Increment the common x axis
					xVal += 0.5;
				}
			};
			//Draw image on canvas
			context.drawImage(video, 0, 0, 192, 144);

			//Get pixel values form canvas
			var frame1 = context.getImageData(0, 0, 192, 144).data;
			
			//Uint8ClampedArray to array
			var frame = Array.from(frame1);

			//Making POST request
			var param = 'frame='+frame;
			xhttp.open("POST", 'opencv/', true);
			xhttp.setRequestHeader('Content-type', 'application/x-www-form-urlencoded')
			xhttp.send(param);
		}
	}, 500);
});