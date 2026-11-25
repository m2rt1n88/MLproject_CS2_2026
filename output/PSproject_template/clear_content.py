from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as ET
import posixpath
from collections import Counter

SOURCE = Path(r'G:\微信\xwechat_files\wxid_2hb6m79o5cp322_db18\msg\file\2026-11\PSproject.pptx')
ROOT = Path(__file__).parent
DEST = ROOT / 'PSproject_空白模板.pptx'
NS = {'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def encode(root):
    return ET.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)

with ZipFile(SOURCE) as z:
    parts = {name:z.read(name) for name in z.namelist() if not name.endswith('/')}
counts = Counter()
for name in list(parts):
    if not name.endswith('.xml') or not name.startswith(('ppt/slides/', 'ppt/slideLayouts/', 'ppt/slideMasters/', 'ppt/notesMasters/', 'ppt/handoutMasters/')):
        continue
    root = ET.fromstring(parts[name])
    for text in root.xpath('//a:t', namespaces=NS):
        if text.text:
            counts['text_runs'] += 1
        text.text = ''
    # Remove every slide-specific image, equation, embedded chart/table/object,
    # including alternate representations. Backgrounds and master art are retained.
    if name.startswith('ppt/slides/'):
        for node in root.xpath('//*[local-name()="AlternateContent"] | //p:sp[p:spPr/a:prstGeom[@prst="ellipse"]]', namespaces=NS):
            node.getparent().remove(node)
            counts['annotations_or_equations'] += 1
        for node in root.xpath('//p:pic | //p:graphicFrame | //p:contentPart | //p:oleObj | //*[local-name()="oMathPara"] | //*[local-name()="oMath"]', namespaces=NS):
            if node.getparent() is not None:
                node.getparent().remove(node)
                counts['content_objects'] += 1
    for node in root.xpath('//a:fld | //a:hlinkClick | //a:hlinkMouseOver | //p:timing | //p:extLst', namespaces=NS):
        if node.getparent() is not None:
            node.getparent().remove(node)
    for paragraph in root.xpath('//a:p', namespaces=NS):
        props = paragraph.find('a:pPr', NS)
        if props is None:
            props = ET.Element('{'+NS['a']+'}pPr')
            paragraph.insert(0, props)
        for node in list(props):
            if ET.QName(node).localname.startswith('bu'):
                props.remove(node)
        ET.SubElement(props, '{'+NS['a']+'}buNone')
    for node in root.xpath('//p:cNvPr', namespaces=NS):
        node.attrib.pop('descr', None)
        node.attrib.pop('title', None)
    parts[name] = encode(root)

# Remove notes, comments, external links and screenshot relationships, then
# remove unreferenced parts so source data does not remain in the ZIP package.
for name in list(parts):
    if not name.endswith('.rels'):
        continue
    relroot = ET.fromstring(parts[name])
    owner = '' if name == '_rels/.rels' else posixpath.join(posixpath.dirname(posixpath.dirname(name)), posixpath.basename(name)[:-5])
    ownerroot = ET.fromstring(parts[owner]) if owner in parts and owner.endswith('.xml') else None
    used_ids = {v for node in ownerroot.iter() for k,v in node.attrib.items() if k.startswith('{'+NS['r']+'}')} if ownerroot is not None else set()
    for rel in list(relroot):
        kind = rel.get('Type','').rsplit('/',1)[-1]
        remove = kind in {'notesSlide','commentAuthors','comments','person','threadedComment','custom-properties'} or rel.get('TargetMode') == 'External'
        if owner.startswith('ppt/slides/') and kind in {'image','chart','oleObject','package','audio','video','media'} and rel.get('Id') not in used_ids:
            remove = True
        if remove:
            relroot.remove(rel)
    parts[name] = encode(relroot)

core = 'docProps/core.xml'
if core in parts:
    root = ET.fromstring(parts[core])
    for node in list(root):
        root.remove(node)
    parts[core] = encode(root)

reachable = {'[Content_Types].xml'}
def visit(owner):
    relname = '_rels/.rels' if not owner else posixpath.join(posixpath.dirname(owner),'_rels',posixpath.basename(owner)+'.rels')
    if relname not in parts or relname in reachable:
        return
    reachable.add(relname)
    for rel in ET.fromstring(parts[relname]):
        if rel.get('TargetMode') == 'External':
            continue
        target = rel.get('Target','')
        target = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(posixpath.dirname(owner),target))
        if target in parts and target not in reachable:
            reachable.add(target)
            visit(target)
visit('')
parts = {k:v for k,v in parts.items() if k in reachable}
ct = ET.fromstring(parts['[Content_Types].xml'])
for node in list(ct):
    if node.get('PartName') and node.get('PartName').lstrip('/') not in parts:
        ct.remove(node)
parts['[Content_Types].xml'] = encode(ct)
with ZipFile(DEST,'w',ZIP_DEFLATED) as z:
    for name,data in parts.items():
        z.writestr(name,data)
print(dict(counts))
print('Retained media:', [n for n in parts if n.startswith('ppt/media/')])
print('Saved:',DEST)
