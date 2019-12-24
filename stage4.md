---
layout: default
---

<div class="posts">
  {% for post in site.stage123 %}
    <article class="post">
      <div class="post_header">
      {% if post.image %}
      <img src="/assets/images/{{post.image}}" alt="thumbnail image">
      {% endif %}
      <h1><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h1>
      </div>

      <div class="entry">
        {{ post.excerpt }}
      </div>

      <a href="{{ site.baseurl }}{{ post.url }}" class="read-more">Read More</a>
    </article>
  {% endfor %}
</div>
