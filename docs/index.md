# 🥔 TATERS: Takes All Things, Extracts Relevant Stuff

<div style="display:flex; align-items:center; justify-content:center;
            gap:1.25rem; flex-wrap:wrap; margin:1.5rem 0;">
  <img src="img/TATERS-small.png" alt="TATERS" width="200"
       style="flex:0 0 auto; max-width:45%; height:auto;">
  <video src="img/TATERS-animation.mp4" autoplay loop muted playsinline
         width="548"
         style="flex:1 1 340px; max-width:548px; height:auto; border-radius:6px;"
         aria-label="TATERS running a pipeline, from picking a source through to the finished report">
    <a href="img/TATERS-animation.mp4">Watch TATERS run a pipeline</a>
  </video>
</div>

---

Available on [GitHub](https://github.com/ryanboyd/taters) and [PyPI](https://pypi.org/project/taters/).

> Status: **early and evolving**. It already works for many common workflows,
> but expect the occasional rough edge and renaming as things mature. Pin a
> version if you need stability.

---

## The backstory

Today, there are more ways to think about language as data than ever before
in human history. The rapid expansion of text analytics and natural language
processing into the social sciences has been great for us "word nerds" with
programming backgrounds, but not so great for curious scholars and students
who are lacking in technical expertise. For people who want a simple way
to get up and running, the idea of learning a new programming language,
figuring out entire ecosystems of package dependencies, and trying to get
things to "just work" can be daunting. Even for those of us who do this for a
living, it can be frustrating at times.

So, I spent years (and years, and years) building lots of little "point, click,
voila!" applications to do different tasks, trying to lower the barrier to
entry for colleagues and newcomers. But, even for me, it was exhausting to
run my data through not one, not two, but many different applications to do
more advanced analyses. I tried o solve *that* problem with a program called
[BUTTER](https://www.butter.tools) — a more comprehensive application where you 
could build your own pipelines without writing a single line of code. 
The idea was simple: there are a lot of methods that social scientists commonly
need, so why not put them all in one place? BUTTER was a good idea with a
lot of shortcomings: it was written in native C#, which meant Windows-only,
hard to extend, and forever cut off from the scientific ecosystem where all of
the interesting breakthroughs were (and still are) happening.

So, here we go again. This time we're baking **TATERS**: written in Python, natively
multi-platform, and it gets things done with gusto. I have to say, so far, I
like it.

## What it does

TATERS turns raw data — video, audio, or text — into analysis-ready
features. And, depending on what your data looks like, hey, it'll even
do some analyses for you. Let's say that you have a dataset of song lyrics
and you want to know how the language changed over the years. Oh me, oh my,
that sounds like you're going to have to do some natural language processing.
Oh, wait, here we go, let's just tell TATERS where the data is, which columns
we care about, and let it do the work for us. What if we have a folder of
PDF files and want readability scores for each one? What if we have a CSV file
of social media posts and want to run a topic model, measure lexical complexity,
calculate some dictionary-based scores per user, and see which of those are
best at predicting a person's age? Yep, just tell TATERS what you need and
it'll do the heavy lifting for you.

TATERS asks what your data is and what you want out of it, works out the steps
to get there, and runs them. What it leaves behind is an ordinary pipeline
file you can re-run, edit, or hand to a colleague — see
[the app guide](guides/wizard.md).

Importantly, the goal is to make it so that you do not have to write any code.
If you *do* write code, every part of **TATERS** is also extremely modular,
which means that you can take any part of it that you want, import it into your
own Python codebase, and run any piece that you want, any way that you want.

The [guides](guides/guides-overview.md) explain what each method is for, and the
[API reference](api/api-overview.md) documents every parameter.

## What it isn't

It isn't edible. It also isn't a replacement for actually *understanding* the methods
that you're using. I'll do my best to point users/readers to what I feel are some
of the more valuable resources and citations for learning more about the rationale
and nuances of different methods. There are a lot of great ideas out there, and it's
a fast-moving field right now, so it's important to plug into the wider world of
scientific discovery if you want to truly get a sense of what you're measuring, why
you're measuring things a certain way, and how it all fits together.