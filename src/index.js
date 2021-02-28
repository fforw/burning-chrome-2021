// noinspection ES6UnusedImports
import STYLE from "./style.css"
import perfNow from "performance-now"

import vertexShaderSource from "./rm-06.vert"
import fragmentShaderSource from "./rm-06.frag"
import Color from "./Color";

//console.log(fragmentShaderSource)

const PHI = (1 + Math.sqrt(5)) / 2;
const TAU = Math.PI * 2;
const DEG2RAD_FACTOR = TAU / 360;

const config = {
    width: 0,
    height: 0
};

let canvas, gl, vao, program;


// uniform: current time
let u_time;

let u_resolution;

let u_mouse;

let u_palette;

let u_shiny;

let mouseX = 0, mouseY = 0, mouseDown, startX, startY;

// Get the container element's bounding box
let canvasBounds;

function resize()
{
    const width = (window.innerWidth) & ~15;
    const height = (window.innerHeight) | 0;

    config.width = width;
    config.height = height;

    canvas.width = width;
    canvas.height = height;

    mouseX = width/2;
    mouseY = height/2;

    gl.viewport(0, 0, canvas.width, canvas.height);
}

function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    const success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
    if (success) {
        return shader;
    }

    console.error(gl.getShaderInfoLog(shader));  // eslint-disable-line
    gl.deleteShader(shader);
    return undefined;
}

function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    const success = gl.getProgramParameter(program, gl.LINK_STATUS);
    if (success) {
        return program;
    }

    console.error(gl.getProgramInfoLog(program));  // eslint-disable-line
    gl.deleteProgram(program);
    return undefined;
}


function printError(msg)
{
    document.getElementById("out").innerHTML = "<p>" + msg + "</p>";
}

function main(time)
{
    const f = mouseDown ? 1 : -1;

    // update uniforms
    gl.uniform1f(u_time, perfNow() / 1000.0);
    gl.uniform2f(u_resolution, config.width, config.height);
    gl.uniform4f(u_mouse, mouseX, config.height - mouseY, startX * f, (config.height - startY) * f);

    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    // draw
    const primitiveType = gl.TRIANGLES;
    const offset = 0;
    const count = 6;
    gl.drawArrays(primitiveType, offset, count);

    requestAnimationFrame(main);
}


window.onload = () => {
    // Get A WebGL context
    canvas = document.getElementById("screen");
    gl = canvas.getContext("webgl2");
    if (!gl) {
        canvas.parentNode.removeChild(canvas);
        printError("Cannot run shader. Your browser does not support WebGL2.");
        return;
    }

    // create GLSL shaders, upload the GLSL source, compile the shaders
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);

    // Link the two shaders into a program
    program = createProgram(gl, vertexShader, fragmentShader);

    // look up where the vertex data needs to go.
    const positionAttributeLocation = gl.getAttribLocation(program, "a_position");

    // Create a buffer and put three 2d clip space points in it
    const positionBuffer = gl.createBuffer();

    // Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

    const positions = [
        -1, -1,
         1, -1,
        -1, 1,
        -1, 1,
         1, 1,
         1,-1
    ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    // Create a vertex array object (attribute state)
    vao = gl.createVertexArray();

    // and make it the one we're currently working with
    gl.bindVertexArray(vao);

    // Turn on the attribute
    gl.enableVertexAttribArray(positionAttributeLocation);

    // Tell the attribute how to get data out of positionBuffer (ARRAY_BUFFER)
    const size = 2;          // 2 components per iteration
    const type = gl.FLOAT;   // the data is 32bit floats
    const normalize = false; // don't normalize the data
    const stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
    let offset = 0;        // start at the beginning of the buffer
    gl.vertexAttribPointer(
        positionAttributeLocation, size, type, normalize, stride, offset);

    resize();

    // Tell WebGL how to convert from clip space to pixels
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);


    u_time = gl.getUniformLocation(program, "u_time");
    u_resolution = gl.getUniformLocation(program, "u_resolution");
    u_mouse = gl.getUniformLocation(program, "u_mouse");
    u_palette = gl.getUniformLocation(program, "u_palette");
    u_shiny = gl.getUniformLocation(program, "u_shiny");

    // Tell it to use our program (pair of shaders)
    gl.useProgram(program);

    // Bind the attribute/buffer set we want.
    gl.bindVertexArray(vao);

    window.addEventListener("resize", resize, true);
    canvas.addEventListener("mousemove", onMouseMove, true);
    canvas.addEventListener("mousedown", onMouseDown, true);
    document.addEventListener("mouseup", onMouseUp, true);

    canvasBounds = document.getElementById("screen").getBoundingClientRect();

    const paletteArray = Color.from(
        [
            "#000",
            "#fff",
            "#c02",
            "#00244f",
            "#004d9d",
            "#010101",
            "#4c3a25",
            "#f0f"
        ],
        1
    );


    gl.uniform3fv(u_palette, paletteArray);
    gl.uniform1fv(u_shiny, new Float32Array([
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10
    ]));

    requestAnimationFrame(main)
}



// Apply the mouse event listener

function onMouseMove(ev)
{
    if (mouseDown)
    {
        mouseX = (ev.clientX - canvasBounds.left) + self.pageXOffset;
        mouseY = (ev.clientY - canvasBounds.top) + self.pageYOffset;
    }
}

function onMouseDown(ev)
{
    mouseDown = true;
    startX = (ev.clientX - canvasBounds.left) + self.pageXOffset;
    startY = (ev.clientY - canvasBounds.top) + self.pageYOffset;
    mouseX = startX;
    mouseY = startY;
}

function onMouseUp(ev)
{
    mouseDown = false;
}

