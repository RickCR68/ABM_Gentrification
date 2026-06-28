<template>
  <div ref="container" :style="style">
    <jupyter-widget
      v-for="child in children"
      :key="child"
      :widget="child"
    />
  </div>
</template>

<script>
module.exports = {
  mounted() {
    this.resizeObserver = new ResizeObserver(() => this.reportSize())
    this.resizeObserver.observe(this.$refs.container)
    this.reportSize()
  },
  beforeDestroy() {
    if (this.resizeObserver) this.resizeObserver.disconnect()
  },
  methods: {
    reportSize() {
      const element = this.$refs.container
      if (!element) return
      this.view_data = {
        width: element.clientWidth,
        height: element.clientHeight
      }
    }
  }
}
</script> 
