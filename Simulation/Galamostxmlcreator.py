#!/usr/bin/python3
#Load the xml library
import xml.dom.minidom
import numpy as np
from io import StringIO

class GalamostXmlCreator(object):

    def __init__(self, filename, **kwargs):
        self.filename=filename;
        self.doc = xml.dom.minidom.Document()
        self.root = self.doc.createElement('galamost_xml')
        self.root.setAttribute('version','1.3')
        self.doc.appendChild(self.root)  
        
        self.configurationTag=self.doc.createElement('configuration')
        self.configurationTag.setAttribute('time_step',"0")
        self.configurationTag.setAttribute('dimensions',"3")
        
        self.root.appendChild(self.configurationTag)  
        
        self.boxTag=self.doc.createElement('box');
        self.configurationTag.appendChild(self.boxTag)  
        
        self.positionTag=self.doc.createElement('position');
        self.configurationTag.appendChild(self.positionTag)  


        self.typeTag=self.doc.createElement('type');
        self.configurationTag.appendChild(self.typeTag)


        

        
    def setbox(self, box):
        self.boxTag.setAttribute('lx','{:.8f}'.format(box[0]));
        self.boxTag.setAttribute('ly','{:.8f}'.format(box[1]));
        self.boxTag.setAttribute('lz','{:.8f}'.format(box[2]));
    
    def add_posdata(self, pos_data):
        tmp=StringIO();
        tmp.write("\n")
        for i in range(pos_data.shape[0]):
            tmp.write(" {:16.10f} {:16.10f} {:16.10f}\n".format(pos_data[i][0],pos_data[i][1],pos_data[i][2]))
        self.positionTag.setAttribute('num',str(pos_data.shape[0]));
        tmp.seek(0)
        self.positionTag.appendChild(self.doc.createTextNode(tmp.read()).cloneNode(True))
        self.configurationTag.setAttribute('natoms',str(pos_data.shape[0]))
        tmp.close()
        
    def add_typedata(self, type_data):
        tmp=StringIO();
        tmp.write("\n")
        for i in range(type_data.shape[0]):
            tmp.write("{}\n".format(type_data[i]))
        self.typeTag.setAttribute('num',str(type_data.shape[0]));
        tmp.seek(0)
        self.typeTag.appendChild(self.doc.createTextNode(tmp.read()).cloneNode(True))
        tmp.close()
        
    def add_bodydata(self, body_data):
        self.bodyTag=self.doc.createElement('body');
        self.configurationTag.appendChild(self.bodyTag);
        tmp=StringIO();
        tmp.write("\n")
        for i in range(body_data.shape[0]):
            tmp.write("{}\n".format(int(body_data[i])))
        self.bodyTag.setAttribute('num',str(body_data.shape[0]));
        tmp.seek(0)
        self.bodyTag.appendChild(self.doc.createTextNode(tmp.read()).cloneNode(True))
        tmp.close()

    def add_h_crisdata(self, cris_data):
        self.h_crisTag=self.doc.createElement('h_cris');
        self.configurationTag.appendChild(self.h_crisTag);
        tmp=StringIO();
        tmp.write("\n")
        for i in range(cris_data.shape[0]):
            tmp.write("{}\n".format(int(cris_data[i])))
        self.h_crisTag.setAttribute('num', str(cris_data.shape[0]));
        tmp.seek(0)
        self.h_crisTag.appendChild(self.doc.createTextNode(tmp.read()).cloneNode(True))
        tmp.close()

    def add_h_initdata(self, h_init_data):
        self.h_initTag=self.doc.createElement('h_init');
        self.configurationTag.appendChild(self.h_initTag);
        tmp=StringIO();
        tmp.write("\n")
        for i in range(h_init_data.shape[0]):
            tmp.write("{}\n".format(int(h_init_data[i])))
        self.h_initTag.setAttribute('num', str(h_init_data.shape[0]));
        tmp.seek(0)
        self.h_initTag.appendChild(self.doc.createTextNode(tmp.read()).cloneNode(True))
        tmp.close()

    def add_orientationdata(self, orientation_data):
        self.orientationTag=self.doc.createElement('quaternion');
        self.configurationTag.appendChild(self.orientationTag);
        tmp=StringIO();
        tmp.write("\n")
        for i in range(orientation_data.shape[0]):
            tmp.write(" {:16.10f} {:16.10f} {:16.10f} {:16.10f}\n".format(orientation_data[i][0],orientation_data[i][1],orientation_data[i][2],orientation_data[i][3]))
        self.orientationTag.setAttribute('num',str(orientation_data.shape[0]));
        tmp.seek(0)
        self.orientationTag.appendChild(self.doc.createTextNode(tmp.read()).cloneNode(True))
        tmp.close()
        
    def add_bonddata(self, bond_data):
        self.bondTag=self.doc.createElement('bond');
        self.configurationTag.appendChild(self.bondTag)
        tmp=StringIO();
        tmp.write("\n")
        for i in range(bond_data.shape[0]):
            tmp.write("{} {} {}\n".format(bond_data[i][0],bond_data[i][1],bond_data[i][2]))
        tmp.seek(0)
        self.bondTag.setAttribute('num',str(bond_data.shape[0]));
        self.bondTag.appendChild(self.doc.createTextNode(tmp.read()).cloneNode(True))
        tmp.close()
        
    def add_imagedata(self, image_data):
        self.imageTag=self.doc.createElement('image');
        self.configurationTag.appendChild(self.imageTag)  
        tmp=StringIO();
        tmp.write("\n")
        for i in range(image_data.shape[0]):
            tmp.write("{:d} {:d} {:d}\n".format(int(image_data[i][0]),int(image_data[i][1]),int(image_data[i][2])))
        tmp.seek(0)
        self.imageTag.setAttribute('num',str(image_data.shape[0]));
        self.imageTag.appendChild(self.doc.createTextNode(tmp.read()).cloneNode(True))
        tmp.close()

    def add_massdata(self, mass_data):
        self.massTag=self.doc.createElement('mass');
        self.configurationTag.appendChild(self.massTag)
        tmp=StringIO();
        tmp.write("\n")
        for i in range(mass_data.shape[0]):
            tmp.write("{:16.10f}\n".format(mass_data[i]))
        self.massTag.setAttribute('num',str(mass_data.shape[0]));
        tmp.seek(0)
        self.massTag.appendChild(self.doc.createTextNode(tmp.read()).cloneNode(True))
        tmp.close()

    def add_diameterdata(self, diameter_data):
        self.diameterTag=self.doc.createElement('diameter');
        self.configurationTag.appendChild(self.diameterTag)
        tmp=StringIO();
        tmp.write("\n")
        for i in range(diameter_data.shape[0]):
            tmp.write("{:.10f}\n".format(diameter_data[i]))
        self.diameterTag.setAttribute('num', str(diameter_data.shape[0]));
        tmp.seek(0)
        self.diameterTag.appendChild(self.doc.createTextNode(tmp.read()).cloneNode(True))
        tmp.close()

    def add_velocitydata(self, velocity_data):
        self.velocityTag = self.doc.createElement('velocity');
        self.configurationTag.appendChild(self.velocityTag)
        tmp = StringIO();
        tmp.write("\n")
        for i in range(velocity_data.shape[0]):
            tmp.write("{:16.10f} {:16.10f} {:16.10f}\n".format(float(velocity_data[i][0]), float(velocity_data[i][1]), float(velocity_data[i][2])))
        tmp.seek(0)
        self.velocityTag.setAttribute('num', str(velocity_data.shape[0]));
        self.velocityTag.appendChild(self.doc.createTextNode(tmp.read()).cloneNode(True))
        tmp.close()

    def write_sample(self,filename='1.xml'):
        fp = open(filename, 'w')
        self.doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        fp.close()
