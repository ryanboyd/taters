# Working with a Database

A lot of lab data doesn't live in spreadsheets. It lives in a MySQL or
PostgreSQL database somebody set up years ago, with the texts in one table,
the participants in another, and the outcomes somewhere else again. Taters
can work there directly: read the text and the columns beside it, join the
tables that need joining, and write its results back as tables rather than
files.

This page covers setting up a connection (including a database you reach
by logging in to a machine first, with ssh or PuTTY), running an extraction
straight from tables in a database, choosing where the results go, what
happens to a run's tables when the run is deleted, and the two Wrangle
errands: copying tables out of a database into a spreadsheet (with joins, if
you need them), and copying a spreadsheet in.

---

## Connections

A connection is how Taters gets to a database server: what kind it is,
where it lives, who you are there, and how you log in. You set one up once,
under **Settings → Database connections → Add a connection**, and it is
saved for next time. The password is not saved, ever; more on that below.

A connection is the server and your login, not a database. Most servers
hold several databases, and which one you want is a question about what
you are doing today, so Taters asks it each time you use the connection
(with the one you used last as the default). One connection to the lab's
server is enough for every database your account can see on it.

### What you'll need to know

Four things, and if you don't know one of them, the person who runs the
database does. It is a normal thing to ask them for.

1. **What kind of database it is.** MySQL (or MariaDB, which is the same
   thing for our purposes), PostgreSQL, or a SQLite file. The two server
   kinds need a small driver installed first; Taters tells you the exact
   command if it is missing (`pip install "taters[db]"`, which installs
   both).
2. **Where it is.** Either an address this computer can reach directly, like
   `db.lab.edu`, or a machine you have to log in to first (see the next
   section, because that is the common case in a university).
3. **Your username on the database.** Not necessarily the same as your login
   on the machine.
4. **How you log in.** Usually a password. Taters asks for it, with dots, and
   keeps it only until you close the app.

That's it. Taters logs in as soon as it has those, before saving anything,
so a wrong password or an unreachable server is found right there rather
than at the end of a long run. It says what it is doing at each step
(opening the tunnel, logging in, listing the databases), so a slow server
is never a blank wait, and it reads back which databases your login can
see. The connection is named after the login, `me-db-lab-edu`, unless you
call it something else.

### If you normally log in to a machine first

A lot of lab databases are not on the network at all. The routine is: you
`ssh` to a server with your key, and once you are there you type `mysql`
and you are in. If that is how it works for you, answer *Through an ssh
host* when Taters asks how the database is reached. It then asks:

* **Which ssh host?** Exactly what you type after `ssh` when you log in by
  hand: `you@myserver.example.edu`.
* **Do you log in to that host with a particular key file?** Say no if
  logging in by hand just works; Taters uses the same keys and the same
  setup as your `ssh` does. Say yes only if you normally pass one with
  `-i`, and a file browser opens in your `.ssh` folder so you can pick it
  (the private key, a file like `id_ed25519` or `id_rsa`, never the `.pub`
  one).
* **What is the database's address, as seen from that machine?** Almost
  always `localhost`, which is the default. That is the whole trick: Taters
  logs in to the machine the way you do, and from there the database is
  local.

Taters logs in to the machine right then, before asking anything about the
database: if the machine can't be reached, there is nothing to ask yet, and
a wrong host or port is fixed on the spot. Taters never asks for an ssh
password. If your key needs a passphrase, it has to be loaded in your ssh
agent, the same as it would for any script.

The first time you connect to a machine from a given computer, ssh doesn't
know it yet, and Taters shows you what it found:

> myserver.example.edu is new to this computer. ssh remembers each server's
> key so that a different machine answering to the same name is noticed.
> This is the key it showed:
>
>     ssh-ed25519  SHA256:Vb7n…
>
> Accept myserver.example.edu's key and continue?

Accepting means trusting that this really is that server, so don't just
say yes. The way to know is to compare the fingerprint with one you got
some other way: whoever runs the server can tell you theirs, and PuTTY or
ssh showed it to you the first time you logged in by hand (PuTTY keeps it
under *Connection → SSH → Host keys* in a saved session). If they match,
accept; Taters records it where ssh keeps them (or hands it to PuTTY's
plink each time), and you won't be asked again. If you have nothing to
compare it with, or you have connected to that machine before and the key
has changed, say no and ask whoever runs it, because that is exactly the
situation this check exists for.

If Taters can't fetch the key to show you (no `ssh-keyscan` on this
computer, or the server won't answer it), it offers to run the by-hand
login right there in the window instead: `ssh -p 130 you@myserver.example.edu`,
where ssh itself shows the fingerprint and asks. Answer there, and Taters
carries on when it comes back. Same advice: say yes only if the fingerprint
matches one you got some other way.

If something goes wrong and you want to change one thing, the "What now?"
menu lets you change just that: the ssh host (a forgotten port, say), the
key file, the database's address as seen from the machine, or the
username, without starting over.

**If you use PuTTY.** Pick your `.ppk` file as the key and Taters tunnels
with PuTTY's own command-line client, `plink`, which comes with PuTTY. It
reads the `.ppk` as it is, uses Pageant if your key has a passphrase, and
trusts the machines PuTTY has already connected to -- so log in once with
PuTTY first if you haven't. If PuTTY isn't installed on the computer
running Taters, it says so, and the other way out is PuTTYgen's
*Conversions → Export OpenSSH key*, which writes a copy of the key that
plain `ssh` can read.

**One thing that surprises people.** On the server, `mysql` logs you in
without asking for anything, so it feels like there is no password. There
is one of two things going on, and Taters can find out which for you:

> On that machine the database may already know you: a login saved in your
> home folder, or a local socket that trusts your ssh login. Taters can look;
> it shows and saves nothing it finds.
>
> Look on the ssh host for how the database logs you in? (Y/n)

Say yes and Taters runs one command over your ssh login that reads your own
`~/.my.cnf` (or `~/.pgpass`) and checks for the database's local socket.
Then one of these:

* **A saved login was found.** Your `.my.cnf` holds a username and a
  password, which is what `mysql` was quietly using. Taters offers a new
  way of getting the password, *From your saved login on that host*: it
  reads the file afresh over ssh each session and keeps the password in
  memory until you close the app, so it is never copied to this computer's
  disk and never shown.
* **The database's socket was found.** The database trusts whoever is
  logged in to the machine (socket authentication). Taters points its
  tunnel at the socket instead of a port, and the database sees the
  connection as *you*, so there is no password at all. This is the plain
  OpenSSH path; PuTTY's plink can't forward a socket, and Taters says so.
* **Nothing there.** Taters asks for the password as usual.

You can say no and set things up by hand, in which case the two cases above
are what to look for yourself: `cat ~/.my.cnf` on the server shows a saved
password, and a database that only trusts the local login needs a password
set by whoever runs it before a tunnel to its *port* can use it.

### The password

**Taters never saves it.** Not in the settings file, not in a run's log,
not in the pipeline. You pick how Taters gets it:

* *Ask me each time* — typed with dots, kept in memory until Taters closes.
* *From an environment variable* — `TATERS_DB_PASSWORD_<NAME>`, for scripts.
* *No password* — for servers that trust your login.

That's a deliberate choice rather than caution for its own sake. Settings
files get synced between machines; a password in one is a password in all of
them. Connections to a server are encrypted unless you chose otherwise, and a
connection that was asked to encrypt and finds it cannot is refused rather
than quietly continued. Through an ssh host the tunnel itself is the
encryption.

### When it won't connect

Every failure comes back as a sentence that says what to do, but here is the
short version of the ones that come up:

* **"Could not reach …"** — the server isn't visible from here. If it only
  works on campus, connect to the VPN first.
* **"… did not accept that password"** — wrong password, or (through an ssh
  host) the account has no password yet; see above. Taters asks once more
  and then stops, so the account is never locked.
* **"… answered as an SSH server, not as MySQL"** — the address you gave is
  the machine you log in to, not the database. Set the connection up as
  *Through an ssh host* instead.
* **"ssh could not log in … the key was not accepted"** — your key isn't
  being offered. Check that `ssh you@host` (or `plink -i key.ppk you@host`)
  works from a terminal without asking for anything.
* **"… could not use the key …"** — the key has a passphrase and nothing is
  holding it open. Load it into your ssh agent, or Pageant for a PuTTY key.
* **"… did not finish logging in within 60 seconds"** — the login is stuck
  waiting for something Taters can't type: a key's passphrase (load the key
  into your agent, or Pageant) or a second factor the server asks for on
  screen. The message ends with the program's own last words, which usually
  say which.
* **"… logged in …, but the tunnel … never opened"** — you are in, but the
  server isn't letting a port be forwarded, which is a setting on its side
  (`AllowTcpForwarding`). Whoever runs it can turn that on.
* **"… nothing answered at localhost:3306 on that machine"** — the tunnel
  opened, but the database isn't where Taters looked. The person who runs it
  knows the port.
* **"… does not offer an encrypted connection"** — the server has no
  encryption set up. Taters offers to go without, for a server inside a
  trusted network only.

### Test it

**Test it** doesn't read your account's grants; it tries. It logs in, asks
which database to check (what an account may do differs by database: read
one, make tables in another), then: Can it read? Can it make a table? Can
it drop one it made? Then it says, in a sentence, what that means: a
read-only account can copy tables out and can be the *source* of a run,
but a run's results will have to go to files or to another database.

---

## Running on a database

Pick *Tables in a database* at "What kind of data do you have?". Then:

1. **Which connection?** A saved one, or add one on the spot. Taters logs
   in and asks **Which database?** from the server's own list (a SQLite file
   is its own database, so that question is skipped). If the account can
   only read that database, Taters says so and carries on: the results will
   go to files, or to another database, which it asks about later.
2. **Which tables?** The database's tables arranged under it, each with its
   size and which columns hold text. Tick the database to take every table,
   or open it up and tick the ones that hold what you need.
3. **Which columns do you need?** Those tables' columns on one screen, laid
   out as the column's name, the table it is in, and its type. Each table is
   a heading with its row count; tick a heading to take the whole table.
   Under each column is what the catalog knows: *the table's key*,
   *→ participants.pid* for a link the database declares, *long text*. (In a
   saved pipeline the columns are written the way SQL names them,
   `posts.body`, `participants.age`.) A database with hundreds of columns gets one
   question first — which tables are involved — so the screen stays a screen.
   Tick the text, anything to group rows by, and any outcome or control the
   statistics will need.

That is the whole choice. From it Taters works out the rest and reads it
back:

* **Which table the rows come from.** The one table you picked; or, with
  several, the one the others are linked *from* (posts point at participants,
  so one row per post); or the one with the long text. Only when none of that
  settles it does it ask: "Which table has one row per piece of text?"
* **How the other tables join on.** The link the database declares, in either
  direction; then a column named after the other table (`post_id` against
  `posts.id`); then a same-named key. A table that only links through a third
  table you didn't pick is joined through it. Each join is read back with its
  counts, computed on the server before anything is written:

  ```
  participants joined on posts.pid = participants.pid (the database declares it).
  Every row in posts with a pid matches a row in participants.
  Working table: 12,000 rows — posts' 3 columns, 3 from participants.
  ```

  When nothing links two tables, it asks which column on each side holds the
  same ID. A join that would repeat text rows (one post matching several
  outcome rows) is said very loudly, with the row count before and after, and
  asked about, default No: every copy would be measured, and every statistic
  would count it more than once. Keys that repeat on *both* sides are refused.
* **What identifies each row.** The row table's key, kept whether or not you
  ticked it. There is no "no id" answer on a database: rows can't be numbered
  by their position, because a row added or removed later would quietly put
  one post's numbers on another. A table without a key is asked for one, and
  a column that repeats is refused with the worst offenders named.
* **Which column is the text.** The one long-text column you picked, said
  out loud; asked only when there are several.

After that the questions are the ones a spreadsheet gets: how to combine
rows, which features, which statistics. Every later picker keeps the same
layout, the table in a column of its own beside each name. A column's output
name stays bare unless two tables share a name, when the second becomes
`participants__created_at`.
Grouping counts, row filters and the outcome checks all come from the server
rather than from a file. When the run starts, the joins are checked again
against the tables as they are now, not as they were when the pipeline was
made: a join that has come to repeat keys on both sides stops the run, and
one that repeats rows is noted in the run log with the counts only, never the
keys.

Every text step runs on a database source and writes its tables to the
run's destination: the n-gram frequency lists, the document-term matrices,
the topic models' loadings and scores, word-vector features and neighbors,
fine-tuned predictions, scored models, averaged features. What is not rows
-- a fitted model, its weights, a report, a figure, the topic models'
k-selection evidence -- is a file, and on a database run it goes under
`db_local/<connection>/<table path>/` in the run folder, so a run's folder
still holds everything that was never a table.

## Where the results go

Anything in, anything out. One question, asked once per run:

> Where should the results go?

*This folder, as spreadsheets* writes `features/`, `gathered/` and the
analysis table as CSV files, exactly as always. A connection, and then a
database on it, where the account can create tables gets every table the
run makes instead: the gathered text
(`taters__study7__gathered__texts`), each feature table
(`taters__study7__features__readability`), the metadata and the analysis
table (`taters__study7__stats_results__analysis_table`). The report, the
figures, any fitted model, the run manifest and the run log always stay in
the run folder, because they aren't rows. The finish screen names the tables
and their row counts.

A database source defaults to its own database when the account can write
there; a spreadsheet or a folder of documents defaults to the folder. Either
can be sent the other way, so a one-off CSV can land in the lab database with
its features beside it, and a database can be measured into plain files.

Every table Taters writes is typed from its contents (a column of integers
is stored as integers, an ID like `007` stays text), carries a hidden row
number so it reads back in the order it was written, and is recorded in
`_taters_tables` with a fingerprint of its rows. A re-run finds its own
tables, checks them against the fingerprint and the settings they were made
with, and leaves them alone when nothing changed, the same resume contract
files have always had. A table somebody edited behind Taters' back fails the
check and is remade. A table too wide for the engine (embeddings, a
document-term matrix with thousands of terms) is stored with its measures
packed into one column; nothing reading it through Taters can tell, and the
ID columns stay real columns so the table still joins.

**Deleting a run.** *Settings → Manage saved pipelines → Delete* removes the
pipeline and, if you say so, its folder. A run that wrote into a database
left tables there, and deleting the folder doesn't touch them, so Taters
says how many there are and asks whether to remove them too (default no).
Only the tables the registry says Taters made under that run are offered;
nothing else in the database is ever touched.

---

## Copying tables out (Wrangle data)

*Wrangle data → Tables from a database, out to spreadsheets.* Needs only
permission to read. It goes step by step:

> What do you want out?
>
>   Each table as its own spreadsheet
>   One spreadsheet from several tables, joined

**Each table as its own spreadsheet** is the plain copy: the rows as they
are, one CSV per table, named after it.

1. **Which connection(s)?** Tick one or several of your saved connections
   (or add one). Taters logs in to each right there.
2. **Which database(s)?** One screen, the databases of each server arranged
   under its connection. Tick a connection to take every database on it, or
   open it up and tick the ones you want; a name the login cannot list can
   be typed. A SQLite file is its own database, so a file skips this.
3. **Which tables?** One screen, the tables of each database arranged under
   it. Tick a database to take every table in it, or open it up and tick the
   ones you want; each table shows its size and which columns hold text.
4. **Keep every column of each table?** Yes is the usual answer. No opens
   the column screen, one per database, with everything ticked, and you
   untick what you don't want.
5. **Where should the spreadsheets go?** A folder, picked in the browser.
   From one database the files land in it as `<table>.csv`; from several,
   each database gets a folder of its own inside it, so two databases with
   a `users` table never collide. If any of the files are already there,
   Taters says which and asks before replacing them.

One thing worth saying plainly: a spreadsheet is a plain copy of whatever
was in those tables. Once it is on a laptop it is outside whatever
protections the database and its server give participant data, and keeping
it where your ethics approval says it may live is your call, not the
tool's. Taters only ever writes where you point it, and asks before
replacing anything.

Then a read-back of what is about to be written, and the files:

```
3 table(s) → /home/me/study7:
    study7/users.csv (480 rows, 4 columns)
    study7/posts.csv (12,000 rows, 5 columns)
    archive/waves.csv (2 rows, 2 columns)
```

**One spreadsheet from several tables, joined** is the run's own two
screens on one database -- which tables, then which of their columns -- for
when you want the participants' ages beside each post rather than in a
separate file. Pick the connection, the database, the tables, then:

> Which columns do you want in the spreadsheet?

Columns from several tables are joined for you on the links the database
declares, with the same read-back, computed on the server before anything is
written:

```
users joined on posts.user_id = users.id (the database declares it).
3 of 4 rows in posts match a row in users; 1 do not (e.g. 9).
1 of 3 rows in users have no match in posts (e.g. 3).
created_at is in both tables and differs on 2 matched rows (e.g. 10: 'p1' vs 'u1'); both are kept, as created_at and users__created_at.
Working table: 4 rows — posts' 5 columns, 3 from users.
```

A column that exists in both tables is kept once if the two agree on every
matched row, and otherwise both are kept, the second as `<table>__<column>`,
the same rule the spreadsheet merge uses. A join that repeats rows (one post
matching several outcome rows) is allowed when copying out, and the read-back
says exactly how many rows that makes. Keys that repeat on *both* sides are
refused, with the worst offenders named, because that almost always means a
column is missing from the key. When nothing links two tables, or two tables
could each be the one the rows come from, it asks.

Then the row count and the query it will run, and the file's name. Every
output is UTF-8, comma-separated, and reads straight into the spreadsheet
flows.

## Copying a spreadsheet in

*Wrangle data → A spreadsheet, into a database table.* Needs permission to
create tables.

Pick the connection, the database on it, and the file (the encoding and
separator are worked out from the file, as everywhere in Taters). Name the table — letters, digits and
underscores, and Taters reads back what it had to tidy — and say which column
identifies each row; that column becomes the table's key, so a repeated value
stops the write rather than slipping through. Then it tells you what it is
about to do and asks before doing it:

```
540 rows × 6 columns → participants in lab. age, score are stored as numbers; the rest as text.
Write it?
```

Columns are typed from their contents. A column whose every value is a plain
integer is stored as one; a plain decimal as a double; anything else, and
anything with a leading zero like `007`, stays text, so an ID is never turned
into a number behind your back.

**Taters never replaces a table it didn't make.** Every table it writes is
recorded in a small registry table of its own (`_taters_tables`), and a name
that is already taken by something else is refused outright.

---

## For the Python-minded

```python
from taters import Taters
t = Taters()

t.export_table(source={"connection": "lab", "database": "study7",
                       "from": {"table": "posts", "as": "p"},
                       "join": [{"table": "users", "as": "u", "on": [["p.user_id", "u.id"]]}]},
               out_csv="posts_with_users.csv")
```

The `source` is the same structure the app builds: a connection, which
database on it, a base table, joins on named keys, and optionally the
output columns. It renders to SQL for whichever engine the connection
speaks, from quoted identifiers only. A table can also be named by
address, `db://<connection>/<database>/<table>`, so `db://lab/study7/posts`;
a SQLite file's one database is called `main`.

## Where to go next

- [The Taters app](wizard.md) — the whole menu, including Wrangle data
- [Utilities & Helpers](utilities.md) — merging, stacking and reshaping spreadsheets
