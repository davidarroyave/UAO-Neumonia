#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py

Interfaz gráfica principal para la herramienta de detección rápida de neumonía.
Integra todos los módulos desarrollados y proporciona una interfaz de usuario
amigable para cargar imágenes, realizar predicciones y generar reportes.
"""

from tkinter import *
from tkinter import ttk, font, filedialog, messagebox
from tkinter.messagebox import askokcancel, showinfo, WARNING
import tkcap
import getpass
from PIL import ImageTk, Image
import csv
import pyautogui
import numpy as np
import time
import tensorflow as tf
import os

# Importar módulos propios
from src.data.read_img import read_image_file
from src.data.integrator import predict


class App:
    """
    Clase principal de la aplicación GUI para detección de neumonía.
    """
    
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        
        # Configuración de la ventana
        self.root.geometry("815x560")
        self.root.resizable(0, 0)
        
        # Inicializar variables
        self.array = None
        self.reportID = 0
        self.img1 = None
        self.img2 = None
        self.label = ""
        self.proba = 0
        self.heatmap = None
        
        # Configurar interfaz
        self.setup_fonts()
        self.setup_variables()
        self.setup_widgets()
        self.setup_layout()
        
        # Establecer foco inicial
        self.text1.focus_set()
        
        # Ejecutar bucle principal
        self.root.mainloop()
    
    def setup_fonts(self):
        """Configura las fuentes utilizadas en la interfaz."""
        self.bold_font = font.Font(weight="bold")
    
    def setup_variables(self):
        """Configura las variables de la aplicación."""
        self.ID = StringVar()
        self.result = StringVar()
    
    def setup_widgets(self):
        """Crea todos los widgets de la interfaz."""
        # LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=self.bold_font)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=self.bold_font)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=self.bold_font)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=self.bold_font)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=self.bold_font
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=self.bold_font)
        
        # INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)
        
        # IMAGE DISPLAY BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        
        # RESULT DISPLAY BOXES
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)
        
        # BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )
    
    def setup_layout(self):
        """Posiciona todos los widgets en la ventana."""
        # LABELS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        
        # BUTTONS POSITIONS
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        
        # INPUT BOXES POSITIONS
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        
        # IMAGE BOXES POSITIONS
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)
    
    def load_img_file(self):
        """Carga un archivo de imagen (DICOM, JPG, PNG)."""
        try:
            filepath = filedialog.askopenfilename(
                initialdir="/",
                title="Seleccionar imagen",
                filetypes=(
                    ("DICOM", "*.dcm"),
                    ("JPEG", "*.jpeg"),
                    ("JPG files", "*.jpg"),
                    ("PNG files", "*.png"),
                )
            )
            
            if filepath:
                # Usar el módulo read_img para cargar la imagen
                self.array, img2show = read_image_file(filepath)
                
                if self.array is not None and img2show is not None:
                    # Redimensionar y mostrar imagen
                    self.img1 = img2show.resize((250, 250), Image.LANCZOS)
                    self.img1 = ImageTk.PhotoImage(self.img1)
                    
                    # Limpiar y mostrar imagen
                    self.text_img1.delete("1.0", END)
                    self.text_img1.image_create(END, image=self.img1)
                    
                    # Habilitar botón de predicción
                    self.button1["state"] = "enabled"
                else:
                    messagebox.showerror("Error", "No se pudo cargar la imagen")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar la imagen: {str(e)}")
    
    def run_model(self):
        """Ejecuta el modelo de predicción."""
        try:
            if self.array is None:
                messagebox.showwarning("Advertencia", "No hay imagen cargada")
                return
            
            # Usar el integrator para realizar la predicción
            self.label, self.proba, self.heatmap = predict(self.array)
            
            if self.label is None:
                messagebox.showerror("Error", "Error en la predicción")
                return
            
            # Mostrar imagen con heatmap
            if self.heatmap is not None:
                self.img2 = Image.fromarray(self.heatmap)
                self.img2 = self.img2.resize((250, 250), Image.LANCZOS)
                self.img2 = ImageTk.PhotoImage(self.img2)
                
                # Limpiar y mostrar imagen con heatmap
                self.text_img2.delete("1.0", END)
                self.text_img2.image_create(END, image=self.img2)
            
            # Mostrar resultados
            self.text2.delete("1.0", END)
            self.text2.insert(END, self.label)
            
            self.text3.delete("1.0", END)
            self.text3.insert(END, f"{self.proba:.2f}%")
            
            print(f"Predicción completada: {self.label} ({self.proba:.2f}%)")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error en la predicción: {str(e)}")
                

    def save_results_csv(self):
        """Guarda los resultados en un archivo CSV."""
        try:
            if not self.label:
                messagebox.showwarning("Advertencia", "No hay resultados para guardar")
                return
            
            with open("reports/historial.csv", "a", newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter="-")
                writer.writerow([
                    self.text1.get(),  # ID del paciente
                    self.label,         # Diagnóstico
                    f"{self.proba:.2f}%"  # Probabilidad
                ])
            
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar: {str(e)}")
           
           
    def create_pdf(self):
        """Genera un reporte en PDF combinando la imagen original y el heatmap."""
        try:
            if not hasattr(self, 'array') or self.array is None:
                messagebox.showwarning("Advertencia", "No hay imagen cargada para generar PDF")
                return
            
            if not hasattr(self, 'label') or not self.label:
                messagebox.showwarning("Advertencia", "No hay resultados para generar PDF")
                return
            
            from PIL import ImageDraw, ImageFont
            from datetime import datetime
            import os
            
            # Crear una imagen para el reporte (tamaño A4 aproximado en píxeles)
            report_width, report_height = 1200, 1600
            report_img = Image.new('RGB', (report_width, report_height), 'white')
            draw = ImageDraw.Draw(report_img)
            
            # Intentar cargar fuentes del sistema
            try:
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
                font_subtitle = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except:
                # Usar fuente por defecto si no se pueden cargar las TTF
                font_title = ImageFont.load_default()
                font_subtitle = ImageFont.load_default()
                font_normal = ImageFont.load_default() 
                font_small = ImageFont.load_default()
            
            # Encabezado del reporte
            y_pos = 40
            draw.text((50, y_pos), "REPORTE DE DETECCIÓN DE NEUMONÍA", fill='navy', font=font_title)
            
            # Línea separadora
            y_pos += 50
            draw.line([(50, y_pos), (report_width-50, y_pos)], fill='navy', width=2)
            
            # Información del reporte
            y_pos += 30
            fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            draw.text((50, y_pos), f"Fecha del análisis: {fecha}", fill='black', font=font_normal)
            
            y_pos += 30
            patient_id = self.text1.get() if self.text1.get() else "No especificado"
            draw.text((50, y_pos), f"ID del Paciente: {patient_id}", fill='black', font=font_normal)
            
            # Sección de resultados
            y_pos += 60
            draw.text((50, y_pos), "RESULTADO DEL DIAGNÓSTICO", fill='darkred', font=font_subtitle)
            
            y_pos += 40
            # Determinar color según el resultado
            result_color = 'green' if self.label.lower() == 'normal' else 'red'
            draw.text((50, y_pos), f"Clasificación: {self.label.upper()}", fill=result_color, font=font_title)
            
            y_pos += 40
            confidence_color = 'green' if self.proba > 80 else 'orange' if self.proba > 60 else 'red'
            draw.text((50, y_pos), f"Probabilidad: {self.proba:.2f}%", fill=confidence_color, font=font_subtitle)
            
            # Nivel de confianza
            y_pos += 30
            if self.proba >= 80:
                confidence_text = "Confianza: ALTA"
                conf_color = 'green'
            elif self.proba >= 60:
                confidence_text = "Confianza: MEDIA"
                conf_color = 'orange'
            else:
                confidence_text = "Confianza: BAJA"
                conf_color = 'red'
            
            draw.text((50, y_pos), confidence_text, fill=conf_color, font=font_normal)
            
            # Sección de imágenes
            y_pos += 80
            draw.text((50, y_pos), "ANÁLISIS VISUAL", fill='darkblue', font=font_subtitle)
            
            images_y = y_pos + 40
            
            # Imagen original
            if hasattr(self, 'array') and self.array is not None:
                try:
                    # Convertir array a PIL Image
                    if len(self.array.shape) == 2:  # Escala de grises
                        original_pil = Image.fromarray(self.array, mode='L').convert('RGB')
                    else:
                        original_pil = Image.fromarray(self.array)
                    
                    # Redimensionar manteniendo proporción
                    img_size = 350
                    original_pil.thumbnail((img_size, img_size), Image.LANCZOS)
                    
                    # Calcular posición centrada
                    img_x = 80
                    report_img.paste(original_pil, (img_x, images_y))
                    draw.text((img_x, images_y - 25), "Imagen Original:", fill='black', font=font_normal)
                    
                except Exception as e:
                    draw.text((80, images_y), f"Error al cargar imagen original: {str(e)}", fill='red', font=font_small)
            
            # Imagen con heatmap
            if hasattr(self, 'heatmap') and self.heatmap is not None:
                try:
                    heatmap_pil = Image.fromarray(self.heatmap)
                    heatmap_pil.thumbnail((img_size, img_size), Image.LANCZOS)
                    
                    heatmap_x = 500
                    report_img.paste(heatmap_pil, (heatmap_x, images_y))
                    draw.text((heatmap_x, images_y - 25), "Análisis Grad-CAM:", fill='black', font=font_normal)
                    
                except Exception as e:
                    draw.text((500, images_y), f"Error al cargar heatmap: {str(e)}", fill='red', font=font_small)
            
            # Explicación del Grad-CAM
            explanation_y = images_y + 380
            draw.text((50, explanation_y), "Grad-CAM (Gradient-weighted Class Activation Mapping):", fill='darkblue', font=font_normal)
            explanation_y += 25
            explanation_text = [
                "• Las áreas rojas/amarillas indican regiones que más influyen en el diagnóstico",
                "• Las áreas azules/verdes tienen menor impacto en la decisión del modelo",
                "• Esta visualización ayuda a entender qué partes de la imagen son más relevantes"
            ]
            
            for line in explanation_text:
                draw.text((70, explanation_y), line, fill='black', font=font_small)
                explanation_y += 20
            
            # Disclaimer legal
            disclaimer_y = explanation_y + 40
            draw.line([(50, disclaimer_y), (report_width-50, disclaimer_y)], fill='red', width=1)
            disclaimer_y += 20
            
            draw.text((50, disclaimer_y), "AVISO LEGAL IMPORTANTE", fill='red', font=font_subtitle)
            disclaimer_y += 30
            
            disclaimer_lines = [
                "• Este software es una herramienta de APOYO al diagnóstico médico",
                "• Los resultados deben ser SIEMPRE validados por un profesional médico",
                "• NO debe utilizarse como único criterio para el diagnóstico",
                "• El diagnóstico final es responsabilidad exclusiva del médico tratante",
                "• En caso de emergencia, consulte inmediatamente con un profesional de la salud"
            ]
            
            for line in disclaimer_lines:
                draw.text((70, disclaimer_y), line, fill='darkred', font=font_small)
                disclaimer_y += 18
            
            # Footer
            footer_y = report_height - 60
            draw.text((50, footer_y), "Sistema de Detección de Neumonía - UAO", fill='gray', font=font_small)
            draw.text((50, footer_y + 20), f"Reporte generado automáticamente - ID: {self.reportID:03d}", fill='gray', font=font_small)
            
            # Guardar como PDF
            folder = "reports"
            pdf_filename = f"{folder}/Reporte_Neumonia_{self.reportID:03d}.pdf"
            report_img.save(pdf_filename, "PDF", resolution=150.0, quality=95)
            
            self.reportID += 1
            
            # Mostrar mensaje de éxito con más información
            showinfo(
                title="PDF Generado Exitosamente", 
                message=f"El reporte médico ha sido generado:\n\n"
                       f"Archivo: {pdf_filename}\n"
                       f"Paciente: {patient_id}\n"
                       f"Diagnóstico: {self.label.upper()}\n"
                       f"Probabilidad: {self.proba:.2f}%\n\n"
                       f"El archivo se encuentra en la carpeta actual del proyecto."
            )
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar PDF: {str(e)}")
            import traceback
            print(f"Error detallado en create_pdf:\n{traceback.format_exc()}")
    
    def delete(self):
        """Borra todos los datos de la interfaz."""
        answer = askokcancel(
            title="Confirmación", 
            message="Se borrarán todos los datos.", 
            icon=WARNING
        )
        
        if answer:
            try:
                # Limpiar campos de texto
                self.text1.delete(0, "end")
                self.text2.delete("1.0", "end")
                self.text3.delete("1.0", "end")
                
                # Limpiar imágenes
                self.text_img1.delete("1.0", "end")
                self.text_img2.delete("1.0", "end")
                
                # Resetear variables
                self.array = None
                self.img1 = None
                self.img2 = None
                self.label = ""
                self.proba = 0
                self.heatmap = None
                
                # Deshabilitar botón de predicción
                self.button1["state"] = "disabled"
                
                showinfo(title="Borrar", message="Los datos se borraron con éxito")
            
            except Exception as e:
                messagebox.showerror("Error", f"Error al borrar: {str(e)}")


def main():
    """Función principal de la aplicación."""
    try:
        app = App()
        return 0
    except Exception as e:
        print(f"Error al iniciar la aplicación: {e}")
        return 1


if __name__ == "__main__":
    main()