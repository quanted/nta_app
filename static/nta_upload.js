$( document ).ready( function() {
    _('jobID').value = getID();
    populateData();
});

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function getID(){
    var address_array = window.location.pathname.split('/');
    if(address_array[address_array.length -1].length === 8){
        return address_array[address_array.length -1];
    }
    let chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let str = '';
    for (let i = 0; i < 8; i++) {
        str += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return str;
}

function populateData(){
    var formdata = new FormData();
    formdata.append("jobid", _('jobID').value);
    formdata.append("ms", "ms2");
    var ajax = new XMLHttpRequest();
    ajax.addEventListener('load', function(event){
        console.log("File name data:" + ajax.response);
        data = JSON.parse(ajax.response);
        console.log("Neg data:" + data['Neg'] + " Pos data:" + data['Pos']);
        data['Neg'].forEach( ele => addProgress(ele, "neg", true));
        data['Pos'].forEach( ele => addProgress(ele, "pos", true));
        }, false);
    ajax.open("POST", "/nta/files");
    ajax.setRequestHeader("X-CSRFToken", getCookie("csrftoken"));
    ajax.send(formdata);
}

/*function getFiles(){
    var formdata = new FormData();
    formdata.append("path", "");
    var ajax = new XMLHttpRequest();
    ajax.addEventListener('load', function(event){
        console.log("File name data:" + ajax.response);
        _('FileNames').innerHTML = ajax.response;
        }, false);
    ajax.open("POST", "/nta/files");
    ajax.setRequestHeader("X-CSRFToken", getCookie("csrftoken"));
    ajax.send(formdata);
}*/

function _(el) {
    return document.getElementById(el);
}

function uploadFile(obj) {
    var mode = obj.classList.contains("PosInput") ? "pos" : "neg";
    var file = _(obj.id).files[0];
    var jobID = _("jobID").value;
    reset(obj)
    console.log("Mode:"+ mode + "  JobID: " + jobID + "  File: " + file.name + " Obj. Class: " + obj.className);
    if(_(file.name)){
        return false;
    }
    addProgress(file.name, mode);
    var progressBar = _(file.name+'/progbar');
    var status = _(file.name+'/status');
    var formdata = new FormData();
    formdata.append("filename", file.name);
    formdata.append("filetype", file.name.split(".").pop());
    formdata.append("jobid", _('jobID').value);
    formdata.append("ms", "ms2");
    formdata.append("mode", mode);
    formdata.append("fileUpload", file);
    var ajax = new XMLHttpRequest();
    ajax.upload.addEventListener("progress", event => {
        var percent = (event.loaded / event.total) * 100;
        console.log(progressBar.value);
        progressBar.value = Math.round(percent);
    }, false);
    ajax.addEventListener("load", event => {
        console.log(progressBar.value);
        progressBar.value = Math.round(100);
        status.textContent = 'Upload Complete';
        
    }, false);
    ajax.addEventListener("error", event => {
        status.textContent = 'Upload Failed';
    }, false);
    ajax.addEventListener("abort", event => {
    }, false);
    ajax.open("POST", "/nta/upload");
    ajax.setRequestHeader("X-CSRFToken", getCookie("csrftoken"))
    ajax.send(formdata);
}

function deleteFile(obj) {
    var mode = obj.classList.contains("PosInput") ? "pos" : "neg";
    var file_name = obj.parentNode.id;
    var formdata = new FormData();
    console.log("Name: " + file_name + " Mode: "  + mode + " Job ID: "+ _('jobID').value + " Obj. Class: " + obj.className);
    formdata.append("filename", file_name);
    formdata.append("jobid", _('jobID').value);
    formdata.append("ms", "ms2");
    formdata.append("mode", mode);
    console.log("Form data: " + formdata);
    var ajax = new XMLHttpRequest();
    ajax.addEventListener('load', event => {
        console.log(ajax.response);
        if(ajax.response === 'File Removed'){
                var parent_div = obj.parentNode;
                parent_div.remove();
        }
    });
    ajax.open("POST", "/nta/delete/");
    ajax.setRequestHeader("X-CSRFToken", getCookie("csrftoken"));
    ajax.send(formdata);
}

function createElement(element, attribute, inner) {
    if (typeof element === "undefined") {
        return false;
    }
    if (typeof inner === "undefined") {
        inner = "";
    }
    var el = document.createElement(element);
    if (typeof attribute === "object") {
        for (var key in attribute) {
            el.setAttribute(key, attribute[key]);
        }
    }
    if (!Array.isArray(inner)) {
        inner = [inner];
    }
    for (var k = 0; k < inner.length; k++) {
        if (inner[k].tagName) {
            el.appendChild(inner[k]);
        } else {
        el.appendChild(document.createTextNode(inner[k]));
        }
    }
    return el;
}

function reset(ele){
    ele.value = "";
}

function removeUpload(ele) {
    deleteFile(ele);
    var parent_div = ele.parentNode;
    parent_div.remove();
}

function addProgress(file_name, mode, complete = false) {
    var objClass = (mode == "pos") ? "PosInput" : "NegInput";
    var name = file_name;
    var button = createElement("button", {class: objClass, onclick: "removeUpload(this)"}, ["X"]);
    var status = createElement("div", {class: objClass, id: file_name + "/status"}, 
                    [createElement("progress", {class: objClass, value: "0", max: "100", style: "width:50px;", id: name + "/progbar"}), ""]);
    var format = createElement("br",{});
    var div_container = createElement("div", {class: objClass, id: file_name}, [name, button, status, format]);
    _(objClass).appendChild(div_container);
    if(complete){
        _(name+"/progbar").value = 100;
        _(name+"/status").innerHTML = "Upload Complete";
    }
}
