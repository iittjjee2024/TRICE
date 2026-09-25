"""Check normalisation against the real match groups observed during EDA."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.stdout.reconfigure(encoding="utf-8")

from trice.normalize import (consonant_skeleton, normalize_address, normalize_name)

print("=" * 96)
print("CROSS-SCRIPT SKELETON EQUALITY (the critical test)")
print("=" * 96)
cross = [
    ("ram marketing private limited", "राम मार्केटिंग प्राइवेट लिमिटेड"),
    ("modern finance", "मॉडर्न फाइनेंस"),
    ("aditya properties llp", "आदित्य प्रॉपर्टीज एलएलपी"),
    ("ace foods limited", "एस फूड्स लिमिटेड"),
    ("tamil nadu", "तमिलनाडु"),
]
for latin, deva in cross:
    a = normalize_name(latin)
    b = normalize_name(deva)
    same = a.skeleton == b.skeleton
    print(f"  {'MATCH ' if same else 'DIFFER'}  {latin!r}")
    print(f"            latin core={a.core()!r} skel={a.skeleton!r}")
    print(f"            deva  core={b.core()!r} skel={b.skeleton!r}")

print()
print("=" * 96)
print("NAME NORMALISATION on real variants")
print("=" * 96)
groups = [
    ["Alpaugh Golden Vance LLC", "Alpaugh  Golden", "Alpaugh G0lden  Vance"],
    ["Marnie Baynes Keystone Bnb Inc", "... THE MARNIE BAYNES KEYSTONE BNB INC",
     "Marnie Baynes-Keystone Bnb Inc", "Marnie Baynes Keystone",
     "Mamie Baynes  Keystone Bnb Inc", "Marnie Baynes Keystone Bnb Inc.",
     "Halonex dba Marnie Baynes Keystone Bnb Inc"],
    ["Tejraj Chits Private Limited", "Tejraj Chits Private Ltd", "tejrajchits.com",
     "Tejraj Chits Limited Services"],
    ["Edwards, Hintze & Dougherty", "Edwards, Hintze + Dougherty Co",
     "edwardshintzedougherty.com", "#edwardshintze",
     "Hintze Edwards, Dougherty Services", "EDWARD5, HINTZE & DOUGHERTY"],
    ["Finest (India) World Private Limited", "M/s Finest (India) World Private Ltd",
     "Finest (India) Woerd Private Limited"],
    ["Meta's Foods", "Meta's  Service", "Meta's-Foods", "Quocalo Co dba Meta's Foods"],
    ["Roxie Culbert, D.D.S., PC", "ROXIE CULBERT, PC CENTER", "Roxie Culbert,"],
    ["Safe Atlantic Hcm Group", "Safe Atlantic Group (Partners)", "Safe Atlantic"],
    ["Marina Ecole France Sarl", "SCI Ptit Àmicale", "Fractales Amis Groupe S.A.S"],
    ["<< Team Ecole", "-- Holloway Peak Inc Seafood"],
]
for g in groups:
    print()
    ref = normalize_name(g[0])
    print(f"  REF  {g[0]!r}")
    print(f"       core={ref.core()!r}  legal={sorted(ref.legal)}  skel={ref.skeleton!r}")
    for v in g[1:]:
        p = normalize_name(v)
        shared = set(ref.core_tokens) & set(p.core_tokens)
        flags = []
        if p.is_domain:
            flags.append("domain")
        if p.had_dba:
            flags.append("dba")
        print(f"       {v!r}")
        print(f"         core={p.core()!r} legal={sorted(p.legal)} "
              f"shared={len(shared)}/{len(set(ref.core_tokens))} {' '.join(flags)}")

print()
print("=" * 96)
print("ADDRESS NORMALISATION on real variants")
print("=" * 96)
addr_groups = [
    ["7031 Invitational Drive, Fl 0, Saint Louis, MO",
     "7031 INVITATIONAL DRIVE, SAINT LOUIS, MO"],
    ["1901 Vista Villas Drive, Leander, TX",
     "1901 VISTA VILLAS DR, LEANDER, TX",
     "LEANDER, TX, VISTA VILLAS DRIVE",
     "TX, Leander, 1901 Vista Villas Drive"],
    ["541 14th Street, Newport, OR",
     "FOURTEENTH STREET, NEWPORT, OR",
     "541 FOURTEENTH SAINT, NEWPORT, OR",
     "541 FOURTEENTH ST, NEWPORT, OR"],
    ["B-6, Ananthi Apartments, 13/12, Zakariah Colony, Iii Street, Choolaimedu, Chennai - 600 094., Tamil Nadu",
     "B-6, Ananthi Apartments, 13/12, Zakariah Colony, Iii Street, Chennai - 600 094., TN",
     "Tamil Nadu, Choolaimedu, Chennai - 600 094., Ananthi Apartments, 13/12, Zakariah Colony, Iii Street, B-6"],
    ["Shp No.328, 3Rd, Flr, Prince, Complex Naval Kishore Rd, Lucknow, Uttar Pradesh",
     "HN 823 SHP NO.328, 3RD, FLR, PRINCE, COMPLEX NAVAL KISHORE RD, LCUKNOW"],
    ["84-11 102 Avenue, Ozone Park, NY", "##84-11 102 AVE, OZONE PAKR, NY",
     "84-11-88 102 Ave, Ozone Park, New York"],
    ["", "<NULL>", "  "],
]
for g in addr_groups:
    print()
    ref = normalize_address(g[0])
    print(f"  REF  {g[0][:78]!r}")
    print(f"       postal={ref.postal!r} house={ref.house!r} digits={ref.digits}")
    print(f"       alpha={ref.alpha_tokens}")
    for v in g[1:]:
        p = normalize_address(v)
        jac_d = (len(set(ref.digits) & set(p.digits)) /
                 max(1, len(set(ref.digits) | set(p.digits))))
        jac_a = (len(set(ref.alpha_tokens) & set(p.alpha_tokens)) /
                 max(1, len(set(ref.alpha_tokens) | set(p.alpha_tokens))))
        print(f"       {v[:78]!r}")
        print(f"         postal={p.postal!r} house={p.house!r} "
              f"digitJac={jac_d:.2f} alphaJac={jac_a:.2f} empty={p.is_empty}")
        print(f"         alpha={p.alpha_tokens}")
