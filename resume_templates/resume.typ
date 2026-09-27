// ATS-friendly single-column resume. Data comes from resume.json (tailoring/render.py).
// No tables, columns, icons or images: parsers read it top to bottom as plain text.
#let d = json("resume.json")

#set document(title: d.contact.name + " - Resume", author: d.contact.name)
#set page(paper: "a4", margin: (x: 1.5cm, top: 1.2cm, bottom: 1.2cm))
#set text(font: ("Calibri", "Carlito", "Arial", "Liberation Sans"), size: 10pt, lang: "en")
#set par(leading: 0.45em, spacing: 0.6em)
#set list(indent: 0.4em, body-indent: 0.5em, spacing: 0.42em)
#show link: set text(fill: black)

#let section(title) = {
  v(0.35em)
  block(below: 0.25em, text(size: 11pt, weight: "bold", upper(title)))
  block(below: 0.4em, line(length: 100%, stroke: 0.6pt))
}

#let entry(e) = {
  block(spacing: 0.45em, below: 0.35em)[
    #text(weight: "bold", e.heading)#if e.subheading != none [ | #e.subheading]
    #h(1fr)
    #if e.dates != none { e.dates }
    #if e.location != none or e.description != none [
      \ #text(style: "italic", size: 9.5pt)[#{
        (e.description, e.location).filter(x => x != none).join(" | ")
      }]
    ]
  ]
  if e.bullets.len() > 0 {
    list(..e.bullets.map(b => b.text))
  }
}

#align(center)[
  #text(size: 17pt, weight: "bold", d.contact.name) \
  #v(-0.3em)
  #text(size: 9.5pt, d.contact_lines.join(linebreak()))
]

#if d.summary != none [
  #section("Summary")
  #d.summary
]

#section("Skills")
#for s in d.skills [
  #text(weight: "bold", s.label): #s.items.join(", ") \
]

#section("Experience")
#for e in d.experience { entry(e) }

#if d.projects.len() > 0 [
  #section("Projects")
  #for e in d.projects { entry(e) }
]

#section("Education")
#for e in d.education { entry(e) }
