from clear_content import SOURCE, NS
from zipfile import ZipFile
from lxml import etree as E
with ZipFile(SOURCE) as z:
    for name in ['ppt/slides/slide4.xml','ppt/slides/slide5.xml','ppt/slides/slide6.xml']:
        print(name)
        root = E.fromstring(z.read(name))
        for node in root.findall('.//p:spTree/*',NS):
            print(E.QName(node).localname, node.xpath('.//p:cNvPr/@name', namespaces=NS),node.xpath('.//a:prstGeom/@prst',namespaces=NS),node.xpath('.//a:blip/@r:embed',namespaces=NS), str(node.xpath('.//a:t/text()',namespaces=NS))[:60])
    for name in z.namelist():
        if name.endswith('.rels') and b'image7' in z.read(name):
            print('IMAGE7',name,z.read(name).decode())
