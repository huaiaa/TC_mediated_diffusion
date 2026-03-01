#!/usr/bin/python3
#Load the xml library
import xml.etree.ElementTree as ET
import numpy as np
from io import StringIO
import bz2

class GalamostXmlreader(object):
    def __init__(self, filename,**kwargs):
        self.filename=filename;
        filetype=filename.split('.');
        if(filename.split('.')[-1]=='bz2'):
            a=bz2.open(filename)
            text=a.read();
            self.xmldoc = ET.fromstring(text);
        elif(filename.split('.')[-1]=='xml'):
            self.xmldoc = ET.parse(filename);
        self.configuration=self.xmldoc.find("./configuration");
        self.time_step=self.configuration.attrib['time_step'];
        self.natoms=self.configuration.attrib['natoms'];
        self.box=self.configuration.find('./box');
        self.lx=self.box.attrib['lx'];
        self.ly=self.box.attrib['ly'];
        self.lz=self.box.attrib['lz'];
        
        
        self.time_step=self.configuration.attrib['time_step'];
        
        self.position=self.configuration.find('./position')
        positionfile=StringIO();
        positionfile.write(self.position.text)
        positionfile.seek(0)
        self.positiondata=np.loadtxt(positionfile)
        
        self.type=self.configuration.find('./type')
        
        typefile=StringIO();
        
        typefile.write(self.type.text)
        typefile.seek(0)
        self.typedata=np.loadtxt(typefile,dtype='str')
        
        self.orientation=self.configuration.find('./quaternion')
        
        if(self.orientation is not None):
            orientationfile=StringIO();
            orientationfile.write(self.orientation.text)
            orientationfile.seek(0)
            self.orientationdata=np.loadtxt(orientationfile,dtype='str')
            
        self.image=self.configuration.find('./image')
        if(self.image is not None):
            imagefile=StringIO();
            imagefile.write(self.image.text)
            
            imagefile.seek(0)
            self.imagedata=np.loadtxt(imagefile,dtype='str')

        self.bond = self.configuration.find('./bond')
        if(self.bond is not None):
            bondfile = StringIO();
            bondfile.write(self.bond.text)
            bondfile.seek(0)
            self.bonddata = np.loadtxt(bondfile, dtype='<U8')

        self.body = self.configuration.find('./body')
        if (self.body is not None):
            bodyfile = StringIO();
            bodyfile.write(self.body.text)
            bodyfile.seek(0)
            self.bodydata = np.loadtxt(bodyfile, dtype='<U8')

        self.mass = self.configuration.find('./mass')
        if (self.mass is not None):
            massfile = StringIO();
            massfile.write(self.mass.text)
            massfile.seek(0)
            self.massdata = np.loadtxt(massfile, dtype='<U8')

        self.velocity = self.configuration.find('./velocity')
        if (self.velocity is not None):
            velocityfile = StringIO();
            velocityfile.write(self.velocity.text)
            velocityfile.seek(0)
            self.velocitydata = np.loadtxt(velocityfile, dtype='<U8')
        self.h_cris = self.configuration.find('./h_cris')
        if (self.h_cris is not None):
            h_crisfile = StringIO();
            h_crisfile.write(self.h_cris.text)
            h_crisfile.seek(0)
            self.h_crisdata = np.loadtxt(h_crisfile, dtype='<U8')

        self.h_init = self.configuration.find('./h_init')
        if (self.h_init is not None):
            h_initfile = StringIO();
            h_initfile.write(self.h_init.text)
            h_initfile.seek(0)
            self.h_initdata = np.loadtxt(h_initfile, dtype='<U8')
        # self.angle=self.configuration.find('./angle')
        # anglefile=StringIO();
        # anglefile.write(self.angle.text())
        # anglefile.seek(0)
        # self.angledata=np.loadtxt(anglefile,dtype='str')
        
        
        