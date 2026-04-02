/**
 * SAM3 Simple Point Collector
 * Uses plain HTML5 Canvas instead of Protovis for better compatibility
 * Version: 2025-01-20-v10-FULL-INTERACTION-FIX
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfy.EasySAM3.SimplePointCollector",

    init() {
        api.addEventListener("easysam3_show_image", (event) => {
            const { node_id, bg_image } = event.detail;
            const node = app.graph.getNodeById(node_id);
            
            if (node && node.canvasWidget) {
                const img = new Image();
                img.onload = () => {
                    node.canvasWidget.image = img;
                    const canvas = node.canvasWidget.canvas;
                    canvas.width = img.width;
                    canvas.height = img.height;
                    
                    const nodeWidth = node.size[0] || 400;
                    const availableWidth = nodeWidth - 20; 
                    const aspectRatio = img.height / img.width;
                    const newWidgetHeight = Math.round(availableWidth * aspectRatio);
                    
                    node._isResizing = true; 
                    node.canvasWidget.widgetHeight = newWidgetHeight;
                    
                    if (node.canvasWidget.container) {
                        node.canvasWidget.container.style.height = newWidgetHeight + "px";
                    }
                    
                    node.setSize([nodeWidth, newWidgetHeight + 80]);
                    
                    setTimeout(() => { 
                        node._isResizing = false; 
                    }, 50);
                    
                    node.setDirtyCanvas(true, true);
                    node.redrawCanvas();
                };
                img.src = "data:image/jpeg;base64," + bg_image;
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "EasySAM3PointCollector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; background: #222; overflow: hidden; box-sizing: border-box; margin: 0; padding: 0; display: flex; align-items: center; justify-content: center;";
                
                const infoBar = document.createElement("div");
                infoBar.style.cssText = "position: absolute; top: 5px; left: 5px; right: 5px; z-index: 10; display: flex; gap: 5px; justify-content: space-between; align-items: center; pointer-events: none;";
                container.appendChild(infoBar);

                const pointsCounter = document.createElement("div");
                pointsCounter.style.cssText = "padding: 7px 10px; background: rgba(0,0,0,0.7); color: #fff; border-radius: 4px; font-size: 12px; font-family: monospace;";
                pointsCounter.textContent = "P: 0 N: 0";
                infoBar.appendChild(pointsCounter);

                const btnGroup = document.createElement("div");
                btnGroup.style.cssText = "display: flex; gap: 5px; pointer-events: auto;";
                infoBar.appendChild(btnGroup);

                const clearButton = document.createElement("button");
                clearButton.textContent = "Clear All";
                clearButton.style.cssText = "padding: 6px 12px; background: #d44; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: bold;";
                btnGroup.appendChild(clearButton);

                const continueButton = document.createElement("button");
                continueButton.textContent = "Continue";
                continueButton.style.cssText = "padding: 6px 10px; background: #7eba76; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: bold;";
                btnGroup.appendChild(continueButton);

                const canvas = document.createElement("canvas");
                canvas.width = 400;
                canvas.height = 300;
                canvas.style.cssText = "display: block; max-width: 100%; max-height: 100%; object-fit: contain; cursor: crosshair; margin: 0 auto;";
                container.appendChild(canvas);
                const ctx = canvas.getContext("2d");

                this.canvasWidget = {
                    canvas: canvas,
                    ctx: ctx,
                    container: container,
                    image: null,
                    positivePoints: [],
                    negativePoints: [],
                    hoveredPoint: null,
                    pointsCounter: pointsCounter,
                    widgetHeight: 300
                };

                const widget = this.addDOMWidget("canvas", "customCanvas", container);
                this.canvasWidget.domWidget = widget;

                widget.computeSize = (width) => {
                    return [width, this.canvasWidget.widgetHeight];
                };

                continueButton.addEventListener("click", () => {
                    const pointsData = {
                        positive: this.canvasWidget.positivePoints,
                        negative: this.canvasWidget.negativePoints
                    };
                    fetch("/sam3_point/continue/" + this.id, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ points: pointsData })
                    });
                });

                clearButton.addEventListener("click", (e) => {
                    e.preventDefault();
                    this.canvasWidget.positivePoints = [];
                    this.canvasWidget.negativePoints = [];
                    this.updatePoints();
                    this.redrawCanvas();
                });

                const originalOnResize = this.onResize;
                this.onResize = function (size) {
                    if (originalOnResize) originalOnResize.apply(this, arguments);
                    if (this._isResizing) return;
                    
                    const newWidgetHeight = Math.max(100, size[1] - 80);
                    this.canvasWidget.widgetHeight = newWidgetHeight;
                    container.style.height = newWidgetHeight + "px";
                    this.redrawCanvas();
                };

                canvas.addEventListener("mousedown", (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
                    const clickedPoint = this.findPointAt(x, y);

                    if (e.button === 2) {
                        if (clickedPoint) {
                            if (clickedPoint.type === 'positive') {
                                this.canvasWidget.positivePoints = this.canvasWidget.positivePoints.filter(p => p !== clickedPoint.point);
                            } else {
                                this.canvasWidget.negativePoints = this.canvasWidget.negativePoints.filter(p => p !== clickedPoint.point);
                            }
                        } else {
                            this.canvasWidget.negativePoints.push({ x, y });
                        }
                    } else if (e.button === 0) {
                        if (e.shiftKey) {
                            this.canvasWidget.negativePoints.push({ x, y });
                        } else if (!clickedPoint) {
                            this.canvasWidget.positivePoints.push({ x, y });
                        }
                    }

                    this.updatePoints();
                    this.redrawCanvas();
                });

                canvas.addEventListener("contextmenu", (e) => e.preventDefault());
                canvas.addEventListener("mousemove", (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
                    const hovered = this.findPointAt(x, y);
                    if (hovered !== this.canvasWidget.hoveredPoint) {
                        this.canvasWidget.hoveredPoint = hovered;
                        this.redrawCanvas();
                    }
                });

                this.redrawCanvas();
                this.setSize([Math.max(400, this.size[0] || 400), 400]);
                container.style.height = "300px";

                return result;
            };

            nodeType.prototype.findPointAt = function (x, y) {
                const threshold = 12;
                for (const point of this.canvasWidget.positivePoints) {
                    if (Math.abs(point.x - x) < threshold && Math.abs(point.y - y) < threshold) return { type: 'positive', point };
                }
                for (const point of this.canvasWidget.negativePoints) {
                    if (Math.abs(point.x - x) < threshold && Math.abs(point.y - y) < threshold) return { type: 'negative', point };
                }
                return null;
            };

            nodeType.prototype.updatePoints = function () {
                this.canvasWidget.pointsCounter.textContent = `P: ${this.canvasWidget.positivePoints.length} N: ${this.canvasWidget.negativePoints.length}`;
            };

            nodeType.prototype.redrawCanvas = function () {
                const { canvas, ctx, image, positivePoints, negativePoints, hoveredPoint } = this.canvasWidget;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (image) {
                    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                } else {
                    ctx.fillStyle = "#333";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = "#666";
                    ctx.font = "16px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Click to add points", canvas.width / 2, canvas.height / 2);
                    ctx.fillText("Left-click: Positive (green)", canvas.width / 2, canvas.height / 2 + 25);
                    ctx.fillText("Shift/Right-click: Negative (red)", canvas.width / 2, canvas.height / 2 + 50);
                }

                const draw = (pts, col) => {
                    ctx.fillStyle = col;
                    pts.forEach(p => {
                        const isHovered = hoveredPoint?.point === p;
                        ctx.beginPath();
                        ctx.arc(p.x, p.y, isHovered ? 10 : 7, 0, Math.PI * 2); 
                        ctx.fill();
                        ctx.strokeStyle = "white"; 
                        ctx.lineWidth = 2; 
                        ctx.stroke();
                    });
                };
                draw(positivePoints, "#0f0");
                draw(negativePoints, "#f00");
            };
        }
    }
});