"""Name pools for fictional entities.

Names should read as ordinary real-world names so the corpus does not look like a
puzzle. Organization names are compounds of two word stems (e.g. "Ashbridge"),
which gives a pool of a few thousand brand words, filtered against well-known
real companies.
"""

FIRST_NAMES = """
Aaron Abigail Adrian Aisha Alan Alejandro Alice Amara Amir Ana Andre Anika Arjun Asha
Ben Bianca Caleb Camila Carlos Carmen Chen Chloe Colin Dana Daniel Daria Darius Diego
Dmitri Eden Elena Elias Elif Emeka Emil Erin Ethan Farah Felix Fiona Gabriel Grace
Hana Hassan Helena Hugo Ibrahim Ines Irene Isaac Ivan Jada Jamal Jasper Javier Jin
Joanna Jonah Julia Kai Kamal Karin Keisha Kenji Kira Lars Laura Leah Leon Lina Lucas
Lucia Maya Malik Marco Maria Mateo Megan Mei Mila Miriam Nadia Naomi Nikhil Nina
Noah Nora Olga Omar Owen Pablo Paloma Pedro Priya Quinn Rafael Rahul Rana Rebecca Reza
Rhea Rosa Ruben Ruth Sahil Samir Sara Sean Selin Simon Sofia Sven Talia Tamsin Tara
Teresa Theo Tobias Tomas Uma Valeria Victor Vikram Wanda Wei Xavier Yara Yasmin Yusuf
Zara Zoe Anton Beatriz Cyrus Delia Esme Femi Greta Henrik Imogen Joaquin Katya Lionel
Marisol Nils Oona Petra Ravi Saoirse Tariq Ursula Viktor Wren Yosef Zainab
""".split()

LAST_NAMES = """
Abbott Acosta Adeyemi Ahmed Alvarez Andersen Arora Bailey Banerjee Barros Becker Bishop
Brennan Calder Campos Carver Castillo Chandra Chávez Cho Clarke Coleman Costa Cruz Dahl
Dalton Das Delgado Desai Dixon Doyle Duarte Eriksen Espinoza Farouk Fischer Fleming
Flores Foster Gallo Garner Gill Goh Gomez Grant Gupta Haddad Halvorsen Hanson Hayes
Herrera Holt Hughes Ibarra Iyer Jensen Joshi Kaplan Karimi Keane Khan Kim Klein Kowalski
Kumar Lambert Larsen Lindqvist Lopez Lowe Lund Madsen Mahmoud Marsh Mehta Mendes Molina
Moreno Murphy Nair Navarro Nguyen Novak Nowak Obi Okafor Olsen Ortiz Osei Park Patel
Pereira Petrov Pham Quinn Rahman Ramos Rao Reyes Rivera Roth Rossi Russo Saeed Salazar
Santos Schmidt Sharma Shah Silva Singh Sokolov Soto Stein Suzuki Tan Tanaka Torres Tran
Usman Vance Varga Vega Volkov Wagner Walsh Weber Wolfe Yamada Yilmaz Young Zhang Zhou
Ziegler Moyo Achebe Bakker Corrigan Draper Ellison Fairbanks Gentry Hale Ingram Juarez
Kerr Lacroix Mercer Nakamura Ochoa Pryce Quintero Ridley Sorensen Thornton Ueda
""".split()

_BRAND_A = """
Ash Alder Bay Bell Birch Bright Brook Cedar Clear Cliff Cobalt Copper Crown Dale Deep
East Elder Elm Fair Fern Flint Fox Frost Glen Granite Grey Hart Hawk Hazel Heath High
Holly Iron Ivy Kestrel Lake Lark Linden Long Maple Marsh Mead Mill Moor Moss North
Oak Peak Pine Quarry Raven Reed Ridge River Rock Rowan Rye Sage Salt Sand Slate South
Spring Star Stone Storm Summer Swift Thorn Tide Timber West Wild Willow Wind Winter
Wolf Wren Yew Amber Briar Clay Cove Dusk Ember Gale Harrow Juniper Larch Myrtle Otter
""".split()

_BRAND_B = """
bridge field gate wood crest ford haven mere stead worth wick ley brook hurst mont vale
point line well more holm shaw combe wright fall light water port side view croft hall
keep reach wold fold mark ton thorpe
""".split()

# Compounds that collide with well-known real organizations or very common words.
_BRAND_BLOCKLIST = {
    "Blackstone", "Stonewater", "Bridgewater", "Fairfield", "Northpoint", "Brightview",
    "Southwest", "Northwest", "Eastside", "Westside", "Springfield", "Oakwood", "Pinewood",
    "Highline", "Starlight", "Moonlight", "Lakeview", "Riverside", "Bayside", "Rockwell",
    "Longview", "Clearwater", "Saltwater", "Fairway", "Highland", "Starmark", "Whitewater",
    "Wildfire", "Stonebridge", "Westfield", "Eastgate", "Southgate", "Northgate", "Highgate",
}

BRAND_WORDS = sorted(
    {a + b for a in _BRAND_A for b in _BRAND_B} - _BRAND_BLOCKLIST
)

COMPANY_SUFFIXES = """
Logistics Capital Systems Labs Biotech Analytics Holdings Partners Robotics Energy
Materials Health Software Foods Freight Media Instruments Networks Therapeutics
Industries Group Consulting Security Aerospace Ventures Semiconductors Textiles
""".split()

FIRM_SUFFIXES = ["Assurance", "Audit Partners", "& Co. Accountants", "Advisory LLP", "Accounting Group"]

SCHOOL_PATTERNS = [
    "{b} University",
    "{b} Institute of Technology",
    "{b} College",
    "University of {b}",
    "{b} School of Business",
]

CITIES = """
Tacoma Spokane Boise Tucson Albuquerque Omaha Wichita Tulsa Madison Duluth Fresno
Sacramento Reno Provo Eugene Richmond Raleigh Savannah Knoxville Dayton Akron Toledo
Rochester Syracuse Hartford Worcester Portland Burlington Lansing Fargo Halifax Calgary
Leeds Bristol Glasgow Cork Lyon Porto Graz Ghent Aarhus Bergen Tampere Gdansk Brno
""".split()

ROLES = [
    "Chief Operating Officer", "VP of Engineering", "Head of Finance", "Director of Research",
    "Senior Product Manager", "Operations Manager", "Lead Data Scientist", "General Counsel",
    "Head of Sales", "Procurement Manager", "Staff Engineer", "Director of Partnerships",
    "Program Manager", "Head of People", "Quality Assurance Lead", "Senior Analyst",
]

PROJECT_WORDS = """
Aurora Basalt Beacon Cascade Cinder Comet Delta Ember Falcon Gemini Glacier Harbor Helix
Horizon Lantern Meridian Monsoon Nimbus Obsidian Onyx Orbit Pinnacle Prism Quartz
Radiant Sable Sentinel Sierra Solstice Sparrow Summit Tempest Tundra Vector Vertex Zephyr
Atlas Bastion Cobra Drift Echo Fathom Granite Halcyon Indigo Jetstream Keystone Lodestar
Mosaic Nova Osprey Paragon Quill Rampart Spindle Tessera Umbra Vanguard Waypoint Zenith
""".split()
